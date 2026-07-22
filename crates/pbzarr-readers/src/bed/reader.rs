//! BED single-column reader implementing `ValueReader` over a bgzipped,
//! tabix-indexed BED. Each `read_into` seeks the index for the requested
//! range and expands interval runs into the caller's buffer.

use std::fs::File;
use std::io::BufRead;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::ArrayViewMut2;
use noodles_bgzf as bgzf;
use noodles_core::Position;
use noodles_core::region::Interval;
use noodles_csi::BinningIndex;

use pbzarr::genome::Genome;
use pbzarr::io::{Numeric, ReaderError, Result, ValueReader};

/// Shared, thread-safe reader state: index + name→id map + target genome.
struct Shared {
    path: PathBuf,
    index: noodles_tabix::Index, // tabix reads into an index carrying a header
    ref_ids: std::collections::HashMap<String, usize>,
    column: usize,
    genome: Genome,
}

/// BED single-column reader. Holds a per-thread bgzf handle behind a mutex;
/// `fork()` opens a fresh handle over the same shared index.
pub struct BedReader<T> {
    shared: Arc<Shared>,
    bgzf: Mutex<bgzf::io::Reader<File>>,
    _marker: PhantomData<T>,
}

pub(super) fn open_bgzf(path: &Path) -> Result<bgzf::io::Reader<File>> {
    let file = File::open(path).map_err(|source| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bgzf::io::Reader::new(file))
}

/// Read the `#`-prefixed header line and return the 0-based index of `name`.
pub fn column_index_by_name<P: AsRef<Path>>(bed_gz: P, name: &str) -> Result<usize> {
    let path = bed_gz.as_ref();
    let mut reader = open_bgzf(path)?;
    let mut line = Vec::new();
    reader
        .read_until(b'\n', &mut line)
        .map_err(|source| ReaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let line = String::from_utf8_lossy(&line);
    let header = line.trim_start_matches('#').trim_end();
    header.split('\t').position(|c| c == name).ok_or_else(|| {
        ReaderError::Other(anyhow::anyhow!(
            "column {name:?} not in header of {}",
            path.display()
        ))
    })
}

impl<T> BedReader<T>
where
    T: Numeric + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    /// Open a bgzipped, tabix-indexed BED. `column` is the absolute 0-based
    /// value column (`>= 3`). `genome` is the target genome (returned by
    /// `contigs()`); its contig names must match the BED's.
    pub fn open<P: AsRef<Path>>(bed_gz: P, column: usize, genome: Genome) -> Result<Self> {
        let path = bed_gz.as_ref().to_path_buf();
        let tbi = PathBuf::from(format!("{}.tbi", path.display()));
        let index = noodles_tabix::fs::read(&tbi).map_err(|source| ReaderError::Io {
            path: tbi.clone(),
            source,
        })?;
        let header = index.header().ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "tabix index {} has no header",
                tbi.display()
            ))
        })?;
        let ref_ids = header
            .reference_sequence_names()
            .iter()
            .enumerate()
            .map(|(i, n)| (String::from_utf8_lossy(n.as_ref()).into_owned(), i))
            .collect();

        let bgzf = open_bgzf(&path)?;
        Ok(Self {
            shared: Arc::new(Shared {
                path,
                index,
                ref_ids,
                column,
                genome,
            }),
            bgzf: Mutex::new(bgzf),
            _marker: PhantomData,
        })
    }
}

impl<T> ValueReader for BedReader<T>
where
    T: Numeric + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    type Item = T;

    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn n_fields(&self) -> usize {
        1
    }

    fn read_into(
        &self,
        contig_name: &str,
        start: u64,
        end: u64,
        mut dst: ArrayViewMut2<'_, Self::Item>,
    ) -> Result<()> {
        if end <= start {
            return Ok(());
        }
        let Some(&ref_id) = self.shared.ref_ids.get(contig_name) else {
            return Ok(()); // contig absent from this BED -> leave as caller's fill
        };

        // tabix/csi intervals are 1-based inclusive; our range is 0-based half-open.
        let q_start = Position::try_from(start as usize + 1)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad start {start}: {e}")))?;
        let q_end = Position::try_from(end as usize)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("bad end {end}: {e}")))?;
        let interval = Interval::from(q_start..=q_end);

        let chunks = self
            .shared
            .index
            .query(ref_id, interval)
            .map_err(|source| ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            })?;

        let mut bgzf = self.bgzf.lock().expect("bed reader mutex poisoned");
        let mut line = Vec::new();
        for chunk in chunks {
            bgzf.seek(chunk.start()).map_err(|source| ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            })?;
            while bgzf.virtual_position() < chunk.end() {
                line.clear();
                let n = bgzf
                    .read_until(b'\n', &mut line)
                    .map_err(|source| ReaderError::Io {
                        path: self.shared.path.clone(),
                        source,
                    })?;
                if n == 0 {
                    break;
                }
                if line.first() == Some(&b'#') {
                    continue;
                }
                self.scatter_line(&line, contig_name, start, end, &mut dst)?;
            }
        }
        Ok(())
    }

    fn fork(&self) -> Result<Self> {
        let bgzf = open_bgzf(&self.shared.path)?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            bgzf: Mutex::new(bgzf),
            _marker: PhantomData,
        })
    }
}

impl<T> BedReader<T>
where
    T: Numeric + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    /// Parse one BED line and write its value across the clipped run in `dst`.
    fn scatter_line(
        &self,
        line: &[u8],
        contig_name: &str,
        win_start: u64,
        win_end: u64,
        dst: &mut ArrayViewMut2<'_, T>,
    ) -> Result<()> {
        let text = std::str::from_utf8(line).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "non-utf8 line in {}: {e}",
                self.shared.path.display()
            ))
        })?;
        let mut fields = text.trim_end().split('\t');
        let chrom = fields.next().unwrap_or_default();
        if chrom != contig_name {
            return Ok(()); // tabix bins may overlap neighbors; skip foreign contigs
        }
        let cols: Vec<&str> = std::iter::once(chrom).chain(fields).collect();
        let parse_u64 = |s: &str| {
            s.parse::<u64>().map_err(|e| {
                ReaderError::Other(anyhow::anyhow!(
                    "bad coord {s:?} in {}: {e}",
                    self.shared.path.display()
                ))
            })
        };
        let bed_start = parse_u64(cols.get(1).copied().unwrap_or_default())?;
        let bed_end = parse_u64(cols.get(2).copied().unwrap_or_default())?;

        let lo = bed_start.max(win_start);
        let hi = bed_end.min(win_end);
        if lo >= hi {
            return Ok(());
        }

        let cell = cols.get(self.shared.column).copied().ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "line has no column {} in {}",
                self.shared.column,
                self.shared.path.display()
            ))
        })?;
        let value: T = cell.parse().map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "parse column {} value {cell:?} in {}: {e}",
                self.shared.column,
                self.shared.path.display()
            ))
        })?;

        for pos in lo..hi {
            let idx = (pos - win_start) as usize;
            dst[[idx, 0]] = value;
        }
        Ok(())
    }
}
