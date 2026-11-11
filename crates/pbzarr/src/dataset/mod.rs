//! Reading and source imports for the PBZ root-Dataset convention.
mod metadata;
mod read;
mod reference;
mod write;
use crate::{PbzError, Result};
pub use metadata::{
    Role, Support, ValidationIssue, ValidationReport, ValidationStatus, VariableInfo,
};
pub use reference::{BinInterval, Contig, ContigId, GenomicAxis, Reference, Region};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
pub(crate) use write::DatasetPlan;
use zarrs::storage::{
    ListableStorage, ReadableStorage, ReadableWritableListableStorage, StorePrefix,
};

/// The `pbz:version` value written to, and required of, a root-Dataset store.
pub const FORMAT_VERSION: &str = "0.5";

/// The dimension of the reference arrays.
pub(crate) const CONTIG_DIM: &str = "contig";
pub(crate) const CONTIG_ID: &str = "contig_id";
pub(crate) const CONTIG_LENGTH: &str = "contig_length";
pub(crate) const CONTIG_MD5: &str = "contig_md5";
/// Root arrays that are coordinates by name. Offsets come from a prefix sum
/// of `contig_length`.
pub(crate) const REFERENCE_ARRAYS: [&str; 3] = [CONTIG_ID, CONTIG_LENGTH, CONTIG_MD5];

/// True when `path` is a file that starts with a zip local file header.
pub fn is_zip_store(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    path.is_file()
        && std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .is_ok()
        && magic == *b"PK\x03\x04"
}

/// A metadata catalog of the direct arrays in a root Xarray Dataset.
///
/// Opening reads metadata only. Catalog entries do not guarantee that a
/// variable's encoding is supported or that its values satisfy PBZ invariants.
pub struct Dataset {
    reader: ReadableStorage,
    /// `None` when the Dataset is open read-only.
    writer: Option<ReadableWritableListableStorage>,
    /// The store directory when the storage is a filesystem store.
    dir: Option<PathBuf>,
    /// The archive that [`Dataset::close`] writes, for a store made by
    /// [`Dataset::create_zip`].
    zip: Option<Arc<crate::zip_store::ZipStore>>,
    /// `None` when the storage cannot list its keys.
    lister: Option<ListableStorage>,
    reference_cache: std::sync::OnceLock<Reference>,
    handles: std::sync::Mutex<BTreeMap<String, Arc<read::ArrayHandle>>>,
    root: serde_json::Value,
    variables: BTreeMap<String, VariableInfo>,
}

impl Dataset {
    /// Open a directory store for reads and writes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path.as_ref())
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Self::open_in(Arc::new(storage), Some(path.as_ref().to_path_buf()))
    }
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        Self::open_in(storage, None)
    }
    /// Open a store for reads only. A directory store is listed to find its
    /// variables, as [`Dataset::open`] does. A zip store (see
    /// [`is_zip_store`]) opens through [`Dataset::open_zip`].
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        if is_zip_store(path.as_ref()) {
            return Self::open_zip(path);
        }
        let storage = Arc::new(
            zarrs::filesystem::FilesystemStore::new(path.as_ref())
                .map_err(|e| PbzError::Store(e.to_string()))?,
        );
        Self::open_reader(
            storage.clone(),
            metadata::Nodes::Listed(storage),
            Some(path.as_ref().to_path_buf()),
        )
    }
    /// Open a store for reads only over storage that need not list or write.
    ///
    /// The variable list comes from the root consolidated metadata, so the
    /// store must have it.
    pub fn open_readable(storage: ReadableStorage) -> Result<Self> {
        Self::open_reader(storage, metadata::Nodes::Consolidated, None)
    }
    /// Open a single-file zip store for reads only, through its root
    /// consolidated metadata.
    pub fn open_zip(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let store_error = |e: &dyn std::fmt::Display| {
            PbzError::Store(format!("open zip store {}: {e}", path.display()))
        };
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| store_error(&"the file name is not valid UTF-8"))?;
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let files = zarrs::filesystem::FilesystemStore::new(parent).map_err(|e| store_error(&e))?;
        let key = zarrs::storage::StoreKey::new(name).map_err(|e| store_error(&e))?;
        let zip = Arc::new(
            zarrs_zip::ZipStorageAdapter::new(Arc::new(files), key).map_err(|e| store_error(&e))?,
        );
        let mut dataset = Self::open_readable(zip.clone())?;
        dataset.lister = Some(zip);
        Ok(dataset)
    }
    fn open_reader(
        reader: ReadableStorage,
        nodes: metadata::Nodes,
        dir: Option<PathBuf>,
    ) -> Result<Self> {
        let lister = match &nodes {
            metadata::Nodes::Listed(lister) => Some(lister.clone()),
            metadata::Nodes::Consolidated => None,
        };
        let (root, variables) = metadata::catalog(&reader, nodes)?;
        let mut dataset = Self::new(reader, None, dir, root, variables);
        dataset.lister = lister;
        Ok(dataset)
    }
    fn open_in(storage: ReadableWritableListableStorage, dir: Option<PathBuf>) -> Result<Self> {
        let reader: ReadableStorage = storage.clone();
        let (root, variables) =
            metadata::catalog(&reader, metadata::Nodes::Listed(storage.clone()))?;
        let mut dataset = Self::new(reader, Some(storage.clone()), dir, root, variables);
        dataset.lister = Some(storage);
        Ok(dataset)
    }
    fn new(
        reader: ReadableStorage,
        writer: Option<ReadableWritableListableStorage>,
        dir: Option<PathBuf>,
        root: serde_json::Value,
        variables: BTreeMap<String, VariableInfo>,
    ) -> Self {
        Self {
            reader,
            writer,
            dir,
            zip: None,
            lister: None,
            reference_cache: std::sync::OnceLock::new(),
            handles: std::sync::Mutex::new(BTreeMap::new()),
            root,
            variables,
        }
    }
    /// The write handle, or an error when the Dataset is open read-only.
    pub(crate) fn writable(&self) -> Result<&ReadableWritableListableStorage> {
        self.writer.as_ref().ok_or_else(|| {
            PbzError::Store(
                "this store is open read-only. Open it with Dataset::open to import or append"
                    .into(),
            )
        })
    }
    pub fn variables(&self) -> &BTreeMap<String, VariableInfo> {
        &self.variables
    }
    /// The bytes and the number of stored objects (chunks or shards) of one
    /// root array, without its `zarr.json`. `None` when the storage cannot
    /// list its keys, as for [`Dataset::open_readable`].
    pub fn stored_size(&self, name: &str) -> Result<Option<StoredSize>> {
        let Some(lister) = &self.lister else {
            return Ok(None);
        };
        let store_error = |e: zarrs::storage::StorageError| PbzError::Store(e.to_string());
        let prefix =
            StorePrefix::new(format!("{name}/")).map_err(|e| PbzError::Store(e.to_string()))?;
        let mut size = StoredSize::default();
        for key in lister.list_prefix(&prefix).map_err(store_error)? {
            if key.as_str().ends_with("zarr.json") {
                continue;
            }
            size.objects += 1;
            size.bytes += self
                .reader
                .size_key(&key)
                .map_err(store_error)?
                .unwrap_or(0);
        }
        Ok(Some(size))
    }
}

/// The stored data of one array. See [`Dataset::stored_size`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct StoredSize {
    pub bytes: u64,
    pub objects: u64,
}
