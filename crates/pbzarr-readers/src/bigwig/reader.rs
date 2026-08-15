//! bigWig single-track reader implementing `ValueReader`.
//!
//! Backed by `bigtools::BigWigRead` over a `ReopenableFile`. Each region is read
//! with `BigWigRead::values`, which returns a per-base `Vec<f32>` with `NaN` for
//! positions the file does not cover. Those gaps are mapped to 0, so "no
//! coverage" reads as 0 (the convention for coverage/percent tracks) and an
//! all-gap chunk equals a 0-fill track's fill value, which zarrs elides on write.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use bigtools::BigWigRead;
use bigtools::utils::file::reopen::ReopenableFile;
use ndarray::ArrayViewMut2;

use pbzarr::genome::{Contig, Genome};
use pbzarr::io::{ReaderError, Result, ValueReader};

/// Read a bigWig file's contig list from its header.
///
/// Returns `(name, length)` pairs in the file's chrom order. Used by the Python
/// `PbzStore.from_bigwig` constructor to size a store directly from the source,
/// so callers do not need pybigtools.
pub fn contigs<P: AsRef<Path>>(src: P) -> Result<Vec<(String, u64)>> {
    let path = src.as_ref();
    let reader = BigWigRead::open_file(path)
        .map_err(|e| ReaderError::Other(anyhow::anyhow!("open {}: {e}", path.display())))?;
    Ok(reader
        .chroms()
        .iter()
        .map(|c| (c.name.clone(), c.length as u64))
        .collect())
}

struct Shared {
    path: PathBuf,
    genome: Genome,
}

/// bigWig single-sample reader. Owns one file-backed `BigWigRead`; multi-thread
/// callers should use `fork()` to get a per-thread handle on the same file.
pub struct BigWigReader {
    shared: Arc<Shared>,
    inner: Mutex<BigWigRead<ReopenableFile>>,
}

impl BigWigReader {
    /// Open a bigWig file. Reads the header to build the canonical `Genome`.
    pub fn open<P: AsRef<Path>>(src: P) -> Result<Self> {
        let path = src.as_ref().to_path_buf();
        let inner = BigWigRead::open_file(&path)
            .map_err(|e| ReaderError::Other(anyhow::anyhow!("open {}: {e}", path.display())))?;

        let contigs: Vec<Contig> = inner
            .chroms()
            .iter()
            .map(|c| Contig {
                name: c.name.clone(),
                length: c.length as u64,
            })
            .collect();
        let genome = Genome::new(contigs).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "invalid contig list in {}: {e}",
                path.display(),
            ))
        })?;

        Ok(Self {
            shared: Arc::new(Shared { path, genome }),
            inner: Mutex::new(inner),
        })
    }
}

impl ValueReader for BigWigReader {
    // bigWig stores a single f32 value per base.
    type Item = f32;

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

        // Resolve the name in the bigWig's own genome; the caller's ContigId
        // (if it had one) belongs to a different namespace.
        if self.shared.genome.id(contig_name).is_none() {
            return Err(ReaderError::ContigNotFound {
                path: self.shared.path.clone(),
                contig: contig_name.to_owned(),
            });
        }
        // bigWig's values() is natively 0-based half-open; no conversion
        // needed, only the narrowing cast to its u32 coordinate type.
        let start_u32 = u32::try_from(start).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "bigWig read requires start <= u32::MAX, got {} in {}",
                start,
                self.shared.path.display(),
            ))
        })?;
        let end_u32 = u32::try_from(end).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "bigWig read requires end <= u32::MAX, got {} in {}",
                end,
                self.shared.path.display(),
            ))
        })?;

        let mut inner = self.inner.lock().expect("bigWig reader mutex poisoned");
        // values() returns a per-base Vec<f32> of length (end - start), filling
        // uncovered positions with NaN. Map those gaps to 0 so "no coverage"
        // reads as 0 (the convention for coverage/percent tracks) and matches a
        // 0-fill track: an all-gap chunk becomes all-zero, equals the fill, and
        // is elided on write.
        let values = inner.values(contig_name, start_u32, end_u32).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "read {contig_name}:{start_u32}-{end_u32} in {}: {e}",
                self.shared.path.display(),
            ))
        })?;

        for (idx, v) in values.into_iter().enumerate() {
            dst[[idx, 0]] = if v.is_nan() { 0.0 } else { v };
        }

        Ok(())
    }

    fn fork(&self) -> Result<Self> {
        let reader = BigWigRead::open_file(&self.shared.path).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!("open {}: {e}", self.shared.path.display(),))
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            inner: Mutex::new(reader),
        })
    }
}
