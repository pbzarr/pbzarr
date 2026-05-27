//! D4 single-track reader implementing `ValueReader`.
//!
//! Uses the mmap-backed `d4::D4TrackReader` (default generics
//! `<BitArrayReader, SparseArrayReader<RangeRecord>>`) and reads each region
//! via `.split(Some(chunk_size))` + `ptab.to_codec().decode_block(...)`. The
//! reference for this pattern is `pyd4::load_values_to_buffer` in the upstream
//! d4 repo.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use d4::D4TrackReader;
use d4::ptab::{DecodeResult, Decoder};
use d4::stab::SecondaryTablePartReader;
use ndarray::ArrayViewMut2;

use crate::genome::{Contig, Genome};
use crate::io::error::{ReaderError, Result};
use crate::io::reader::ValueReader;

struct Shared {
    path: PathBuf,
    genome: Genome,
}

/// D4 single-sample reader. Owns one mmap-backed `D4TrackReader`; multi-thread
/// callers should use `fork()` to get a per-thread handle on the same file.
pub struct D4Reader {
    shared: Arc<Shared>,
    inner: Mutex<D4TrackReader>,
}

impl D4Reader {
    /// Open a d4 file. Reads the header to build the canonical `Genome`.
    /// Picks the first available data track via `open_first_track`.
    pub fn open<P: AsRef<Path>>(src: P) -> Result<Self> {
        let path = src.as_ref().to_path_buf();
        let inner = D4TrackReader::open_first_track(&path).map_err(|source| ReaderError::Io {
            path: path.clone(),
            source,
        })?;

        let contigs: Vec<Contig> = inner
            .header()
            .chrom_list()
            .iter()
            .map(|c| Contig {
                name: c.name.clone(),
                length: c.size as u64,
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

impl ValueReader for D4Reader {
    // d4's native dtype is i32 (signed; some signal tracks emit negatives).
    type Item = i32;

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

        // Resolve the name in the d4 file's own genome; the caller's
        // ContigId (if it had one) belongs to a different namespace.
        if self.shared.genome.id(contig_name).is_none() {
            return Err(ReaderError::ContigNotFound {
                path: self.shared.path.clone(),
                contig: contig_name.to_owned(),
            });
        }
        let start_u32 = u32::try_from(start).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "d4 read requires start <= u32::MAX, got {} in {}",
                start,
                self.shared.path.display(),
            ))
        })?;
        let end_u32 = u32::try_from(end).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "d4 read requires end <= u32::MAX, got {} in {}",
                end,
                self.shared.path.display(),
            ))
        })?;
        let span = (end_u32 - start_u32) as usize;

        let mut inner = self.inner.lock().expect("d4 reader mutex poisoned");
        // split() requires &mut self; the partitioning is cheap relative to
        // the actual decode of a 1 Mbp range. Sizing the limit at the request
        // span gives at most one partition per contig overlapping our range.
        let partitions = inner
            .split(Some(span.max(1)))
            .map_err(|source| ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            })?;

        for (mut ptab, mut stab) in partitions {
            let (part_chr, part_begin, part_end) = {
                let (c, b, e) = ptab.region();
                (c.to_string(), b, e)
            };
            if part_chr != contig_name {
                continue;
            }
            let from = start_u32.max(part_begin);
            let to = end_u32.min(part_end);
            if from >= to {
                continue;
            }

            let mut codec = ptab.to_codec();
            codec.decode_block(from as usize, (to - from) as usize, |pos, value| {
                let resolved = match value {
                    DecodeResult::Definitely(v) => v,
                    DecodeResult::Maybe(v) => stab.decode(pos as u32).unwrap_or(v),
                };
                let idx = pos - start_u32 as usize;
                dst[[idx, 0]] = resolved;
            });
        }

        Ok(())
    }

    fn fork(&self) -> Result<Self> {
        let reader = D4TrackReader::open_first_track(&self.shared.path).map_err(|source| {
            ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            }
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            inner: Mutex::new(reader),
        })
    }
}
