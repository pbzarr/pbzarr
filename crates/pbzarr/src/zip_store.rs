//! A write-once storage that builds a single-file zip store.
//!
//! The archive is Zip64 and every entry is STORED. Each data key is written
//! once, as soon as it arrives, with its CRC and sizes in the local header, so
//! the archive has no data descriptors. The `zarr.json` keys stay in memory
//! until [`ZipStore::finish`], which writes them, then the central directory
//! with the root `zarr.json` first.
//!
//! The archive is written to a temporary file beside the target and renamed
//! over the target at finish. A store dropped before finish removes that file.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

use crate::{PbzError, Result};

const ROOT_METADATA: &str = "zarr.json";
const VERSION_NEEDED: u16 = 45;
/// Unix host, zip specification 4.5.
const VERSION_MADE_BY: u16 = (3 << 8) | 45;
const UTF8_NAMES: u16 = 1 << 11;
/// 1980-01-01, the earliest DOS date.
const DOS_DATE: u16 = (1 << 5) | 1;
const FILE_MODE: u32 = 0o100644 << 16;
const MASK32: u32 = u32::MAX;
const ZIP64_EXTRA_ID: u16 = 0x0001;

fn is_metadata(key: &str) -> bool {
    key == ROOT_METADATA || key.ends_with("/zarr.json")
}

fn storage_error(message: String) -> StorageError {
    StorageError::Other(format!("zip store: {message}"))
}

struct Entry {
    name: String,
    crc: u32,
    size: u64,
    /// Offset of the local header.
    offset: u64,
    /// Offset of the entry data.
    data: u64,
}

struct Archive {
    file: BufWriter<File>,
    /// Bytes written so far.
    end: u64,
    entries: Vec<Entry>,
    by_name: HashMap<String, usize>,
    /// Set when a write or a seek fails. The file can then hold bytes that no
    /// entry records, so later writes and finish are refused.
    failed: bool,
}

impl Archive {
    fn check(&self) -> std::io::Result<()> {
        if self.failed {
            return Err(std::io::Error::other(
                "an earlier write to the archive failed",
            ));
        }
        Ok(())
    }

    fn append(&mut self, name: &str, value: &[u8]) -> std::io::Result<()> {
        self.check()?;
        let result = self.write_entry(name, value);
        self.failed |= result.is_err();
        result
    }

    fn write_entry(&mut self, name: &str, value: &[u8]) -> std::io::Result<()> {
        let crc = crc32fast::hash(value);
        let size = value.len() as u64;
        let offset = self.end;
        let mut header = Vec::with_capacity(50 + name.len());
        put32(&mut header, 0x0403_4b50);
        put16(&mut header, VERSION_NEEDED);
        put16(&mut header, UTF8_NAMES);
        put16(&mut header, 0);
        put16(&mut header, 0);
        put16(&mut header, DOS_DATE);
        put32(&mut header, crc);
        put32(&mut header, MASK32);
        put32(&mut header, MASK32);
        put16(&mut header, name_len(name)?);
        put16(&mut header, 20);
        header.extend_from_slice(name.as_bytes());
        put16(&mut header, ZIP64_EXTRA_ID);
        put16(&mut header, 16);
        put64(&mut header, size);
        put64(&mut header, size);
        self.file.write_all(&header)?;
        self.file.write_all(value)?;
        let data = offset + header.len() as u64;
        self.end = data + size;
        self.by_name.insert(name.to_owned(), self.entries.len());
        self.entries.push(Entry {
            name: name.to_owned(),
            crc,
            size,
            offset,
            data,
        });
        Ok(())
    }

    fn read(&mut self, index: usize, start: u64, len: u64) -> std::io::Result<Vec<u8>> {
        self.file.flush()?;
        let at = self.entries[index].data + start;
        let file = self.file.get_mut();
        file.seek(SeekFrom::Start(at))?;
        let mut out = vec![0; len as usize];
        let read = file.read_exact(&mut out);
        if let Err(e) = file.seek(SeekFrom::Start(self.end)) {
            self.failed = true;
            return Err(e);
        }
        read.map(|()| out)
    }

    /// Write the central directory in `order` and the end records.
    fn close(&mut self, order: &[usize]) -> std::io::Result<()> {
        self.check()?;
        let start = self.end;
        let mut directory = Vec::new();
        for &index in order {
            let entry = &self.entries[index];
            put32(&mut directory, 0x0201_4b50);
            put16(&mut directory, VERSION_MADE_BY);
            put16(&mut directory, VERSION_NEEDED);
            put16(&mut directory, UTF8_NAMES);
            put16(&mut directory, 0);
            put16(&mut directory, 0);
            put16(&mut directory, DOS_DATE);
            put32(&mut directory, entry.crc);
            put32(&mut directory, MASK32);
            put32(&mut directory, MASK32);
            put16(&mut directory, name_len(&entry.name)?);
            put16(&mut directory, 28);
            put16(&mut directory, 0);
            put16(&mut directory, 0);
            put16(&mut directory, 0);
            put32(&mut directory, FILE_MODE);
            put32(&mut directory, MASK32);
            directory.extend_from_slice(entry.name.as_bytes());
            put16(&mut directory, ZIP64_EXTRA_ID);
            put16(&mut directory, 24);
            put64(&mut directory, entry.size);
            put64(&mut directory, entry.size);
            put64(&mut directory, entry.offset);
        }
        let count = order.len() as u64;
        let zip64_end = start + directory.len() as u64;
        let mut tail = directory;
        put32(&mut tail, 0x0606_4b50);
        put64(&mut tail, 44);
        put16(&mut tail, VERSION_MADE_BY);
        put16(&mut tail, VERSION_NEEDED);
        put32(&mut tail, 0);
        put32(&mut tail, 0);
        put64(&mut tail, count);
        put64(&mut tail, count);
        put64(&mut tail, zip64_end - start);
        put64(&mut tail, start);
        put32(&mut tail, 0x0706_4b50);
        put32(&mut tail, 0);
        put64(&mut tail, zip64_end);
        put32(&mut tail, 1);
        put32(&mut tail, 0x0605_4b50);
        put16(&mut tail, 0);
        put16(&mut tail, 0);
        put16(&mut tail, u16::MAX);
        put16(&mut tail, u16::MAX);
        put32(&mut tail, MASK32);
        put32(&mut tail, MASK32);
        put16(&mut tail, 0);
        self.file.write_all(&tail)?;
        self.end += tail.len() as u64;
        self.file.flush()?;
        self.file.get_ref().sync_all()
    }
}

fn name_len(name: &str) -> std::io::Result<u16> {
    u16::try_from(name.len())
        .map_err(|_| std::io::Error::other(format!("entry name {name:?} is too long")))
}

fn put16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// A zarrs storage that writes a new zip store once.
///
/// Each data key takes one write. A second write, or an erase of a written
/// data key, is an error; an erase of a key that was never written does
/// nothing. Reads and listings come from the `zarr.json` keys in memory and
/// from the data entries written so far.
pub(crate) struct ZipStore {
    target: PathBuf,
    temporary: PathBuf,
    metadata: Mutex<BTreeMap<String, Bytes>>,
    /// `None` after finish.
    archive: Mutex<Option<Archive>>,
}

impl ZipStore {
    /// Start a zip store that [`ZipStore::finish`] moves to `target`. The
    /// target must not exist.
    pub(crate) fn create(target: &Path) -> Result<Self> {
        if target.exists() {
            return Err(PbzError::Store(format!(
                "{} exists. A zip store is written once, so give a new path",
                target.display()
            )));
        }
        let name = target
            .file_name()
            .ok_or_else(|| PbzError::Store(format!("{} is not a file path", target.display())))?;
        let mut temporary_name = std::ffi::OsString::from(".");
        temporary_name.push(name);
        temporary_name.push(format!(".{}.tmp", std::process::id()));
        let temporary = target.with_file_name(temporary_name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| PbzError::Store(format!("create {}: {e}", temporary.display())))?;
        Ok(Self {
            target: target.to_path_buf(),
            temporary,
            metadata: Mutex::new(BTreeMap::new()),
            archive: Mutex::new(Some(Archive {
                file: BufWriter::new(file),
                end: 0,
                entries: Vec::new(),
                by_name: HashMap::new(),
                failed: false,
            })),
        })
    }

    /// Write the metadata entries and the central directory, then move the
    /// archive to its target. Later writes fail.
    pub(crate) fn finish(&self) -> Result<()> {
        let metadata = self.metadata()?;
        let mut guard = self.archive()?;
        let archive = guard
            .as_mut()
            .ok_or_else(|| PbzError::Store("zip store: the archive is already finished".into()))?;
        if !metadata.contains_key(ROOT_METADATA) {
            return Err(PbzError::Store(
                "zip store: there is no root zarr.json to write".into(),
            ));
        }
        let io = |e: std::io::Error| {
            PbzError::Store(format!(
                "zip store: write {}: {e}",
                self.temporary.display()
            ))
        };
        let first_metadata = archive.entries.len();
        let mut order: Vec<usize> = Vec::with_capacity(first_metadata + metadata.len());
        for (name, value) in metadata.iter() {
            if name == ROOT_METADATA {
                order.insert(0, archive.entries.len());
            } else {
                order.push(archive.entries.len());
            }
            archive.append(name, value).map_err(io)?;
        }
        order.extend(0..first_metadata);
        archive.close(&order).map_err(io)?;
        if self.target.exists() {
            return Err(PbzError::Store(format!(
                "{} exists. A zip store is written once, so give a new path",
                self.target.display()
            )));
        }
        std::fs::rename(&self.temporary, &self.target).map_err(|e| {
            PbzError::Store(format!(
                "zip store: move {} to {}: {e}",
                self.temporary.display(),
                self.target.display()
            ))
        })?;
        *guard = None;
        Ok(())
    }

    fn metadata(&self) -> Result<MutexGuard<'_, BTreeMap<String, Bytes>>> {
        self.metadata
            .lock()
            .map_err(|_| PbzError::Store("zip store: a writer panicked".into()))
    }

    fn archive(&self) -> Result<MutexGuard<'_, Option<Archive>>> {
        self.archive
            .lock()
            .map_err(|_| PbzError::Store("zip store: a writer panicked".into()))
    }

    fn open_archive(&self) -> std::result::Result<MutexGuard<'_, Option<Archive>>, StorageError> {
        let guard = self
            .archive()
            .map_err(|e| StorageError::Other(e.to_string()))?;
        if guard.is_none() {
            return Err(storage_error("the archive is already finished".into()));
        }
        Ok(guard)
    }

    fn keys(&self) -> std::result::Result<BTreeSet<String>, StorageError> {
        let mut keys: BTreeSet<String> = self
            .metadata()
            .map_err(|e| StorageError::Other(e.to_string()))?
            .keys()
            .cloned()
            .collect();
        let guard = self.open_archive()?;
        keys.extend(guard.iter().flat_map(|a| a.by_name.keys().cloned()));
        Ok(keys)
    }
}

impl Drop for ZipStore {
    fn drop(&mut self) {
        if self.archive.get_mut().map(|a| a.is_some()).unwrap_or(true) {
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

fn ranges(
    bytes: Bytes,
    byte_ranges: ByteRangeIterator<'_>,
) -> std::result::Result<Vec<std::result::Result<Bytes, StorageError>>, StorageError> {
    let len = bytes.len() as u64;
    Ok(byte_ranges
        .map(|range| {
            let (start, end) = (range.start(len), range.end(len));
            if start > end || end > len {
                return Err(InvalidByteRangeError::new(range, len).into());
            }
            Ok(bytes.slice(start as usize..end as usize))
        })
        .collect())
}

impl ReadableStorageTraits for ZipStore {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> std::result::Result<MaybeBytesIterator<'a>, StorageError> {
        let Some(value) = self.get(key)? else {
            return Ok(None);
        };
        Ok(Some(Box::new(ranges(value, byte_ranges)?.into_iter())))
    }

    fn get(&self, key: &StoreKey) -> std::result::Result<MaybeBytes, StorageError> {
        if is_metadata(key.as_str()) {
            return Ok(self
                .metadata()
                .map_err(|e| StorageError::Other(e.to_string()))?
                .get(key.as_str())
                .cloned());
        }
        let mut guard = self.open_archive()?;
        let archive = guard.as_mut().expect("open archive");
        let Some(&index) = archive.by_name.get(key.as_str()) else {
            return Ok(None);
        };
        let size = archive.entries[index].size;
        let bytes = archive.read(index, 0, size)?;
        Ok(Some(bytes.into()))
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> std::result::Result<MaybeBytes, StorageError> {
        let Some(value) = self.get(key)? else {
            return Ok(None);
        };
        let mut out = ranges(value, Box::new(std::iter::once(byte_range)))?;
        out.pop().expect("one byte range").map(Some)
    }

    fn size_key(&self, key: &StoreKey) -> std::result::Result<Option<u64>, StorageError> {
        if is_metadata(key.as_str()) {
            return Ok(self
                .metadata()
                .map_err(|e| StorageError::Other(e.to_string()))?
                .get(key.as_str())
                .map(|v| v.len() as u64));
        }
        let guard = self.open_archive()?;
        let archive = guard.as_ref().expect("open archive");
        Ok(archive
            .by_name
            .get(key.as_str())
            .map(|&i| archive.entries[i].size))
    }

    fn supports_get_partial(&self) -> bool {
        false
    }
}

impl WritableStorageTraits for ZipStore {
    fn set(&self, key: &StoreKey, value: Bytes) -> std::result::Result<(), StorageError> {
        if is_metadata(key.as_str()) {
            drop(self.open_archive()?);
            self.metadata()
                .map_err(|e| StorageError::Other(e.to_string()))?
                .insert(key.as_str().to_owned(), value);
            return Ok(());
        }
        let mut guard = self.open_archive()?;
        let archive = guard.as_mut().expect("open archive");
        if archive.by_name.contains_key(key.as_str()) {
            return Err(storage_error(format!(
                "{key} is already written, and each key is written once"
            )));
        }
        archive.append(key.as_str(), &value)?;
        Ok(())
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        _offset_values: OffsetBytesIterator,
    ) -> std::result::Result<(), StorageError> {
        Err(storage_error(format!(
            "{key}: partial writes are not supported"
        )))
    }

    fn erase(&self, key: &StoreKey) -> std::result::Result<(), StorageError> {
        if is_metadata(key.as_str()) {
            self.metadata()
                .map_err(|e| StorageError::Other(e.to_string()))?
                .remove(key.as_str());
            return Ok(());
        }
        let guard = self.open_archive()?;
        if guard.iter().any(|a| a.by_name.contains_key(key.as_str())) {
            return Err(storage_error(format!(
                "{key} is already written and cannot be erased"
            )));
        }
        Ok(())
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> std::result::Result<(), StorageError> {
        {
            let guard = self.open_archive()?;
            let archive = guard.as_ref().expect("open archive");
            if let Some(key) = archive
                .by_name
                .keys()
                .find(|k| k.starts_with(prefix.as_str()))
            {
                return Err(storage_error(format!(
                    "{key} is already written and cannot be erased"
                )));
            }
        }
        self.metadata()
            .map_err(|e| StorageError::Other(e.to_string()))?
            .retain(|k, _| !k.starts_with(prefix.as_str()));
        Ok(())
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

impl ListableStorageTraits for ZipStore {
    fn list(&self) -> std::result::Result<StoreKeys, StorageError> {
        self.keys()?
            .into_iter()
            .map(|k| StoreKey::new(k).map_err(StorageError::from))
            .collect()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> std::result::Result<StoreKeys, StorageError> {
        Ok(self
            .list()?
            .into_iter()
            .filter(|k| k.has_prefix(prefix))
            .collect())
    }

    fn list_dir(
        &self,
        prefix: &StorePrefix,
    ) -> std::result::Result<StoreKeysPrefixes, StorageError> {
        let mut keys = Vec::new();
        let mut prefixes = BTreeSet::new();
        for key in self.keys()? {
            let Some(rest) = key.strip_prefix(prefix.as_str()) else {
                continue;
            };
            match rest.split_once('/') {
                Some((child, _)) => {
                    prefixes.insert(format!("{}{child}/", prefix.as_str()));
                }
                None => keys.push(StoreKey::new(key)?),
            }
        }
        let prefixes = prefixes
            .into_iter()
            .map(|p| StorePrefix::new(p).map_err(StorageError::from))
            .collect::<std::result::Result<_, _>>()?;
        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> std::result::Result<u64, StorageError> {
        let mut size = 0;
        for key in self.list_prefix(prefix)? {
            size += self.size_key(&key)?.unwrap_or(0);
        }
        Ok(size)
    }
}
