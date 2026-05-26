//! D4 single-track reader implementing `ValueReader`.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use d4::ssio::D4TrackReader;
use ndarray::ArrayViewMut2;

use crate::genome::{Contig, Genome, Region};
use crate::io::error::{ReaderError, Result};
use crate::io::reader::ValueReader;

struct Shared {
    path: PathBuf,
    genome: Genome,
}

/// D4 single-sample reader.
pub struct D4Reader {
    shared: Arc<Shared>,
    inner: Mutex<D4TrackReader<File>>,
}

impl D4Reader {
    /// Open a d4 file. Reads the header to build the canonical `Genome`.
    pub fn open<P: AsRef<Path>>(src: P) -> Result<Self> {
        let path = src.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|source| ReaderError::Io {
            path: path.clone(),
            source,
        })?;
        let inner = D4TrackReader::from_reader(file, None).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "failed to open d4 track for {}: {e}",
                path.display(),
            ))
        })?;

        let contigs: Vec<Contig> = inner
            .get_header()
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
    type Item = u32;

    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn n_fields(&self) -> usize {
        1
    }

    fn read_into(&self, region: &Region, mut dst: ArrayViewMut2<'_, Self::Item>) -> Result<()> {
        if region.is_empty() {
            return Ok(());
        }

        let contig =
            self.shared
                .genome
                .get(region.contig)
                .ok_or_else(|| ReaderError::ContigNotFound {
                    path: self.shared.path.clone(),
                    contig: format!("{}", region.contig),
                })?;
        let start = u32::try_from(region.start).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "d4 read requires start <= u32::MAX, got {} in {}",
                region.start,
                self.shared.path.display(),
            ))
        })?;
        let end = u32::try_from(region.end).map_err(|_| {
            ReaderError::Other(anyhow::anyhow!(
                "d4 read requires end <= u32::MAX, got {} in {}",
                region.end,
                self.shared.path.display(),
            ))
        })?;

        let mut inner = self.inner.lock().expect("d4 reader mutex poisoned");
        let mut view = inner.get_view(&contig.name, start, end).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "d4 view failed for {}:{}:{}-{}: {e}",
                self.shared.path.display(),
                contig.name,
                start,
                end,
            ))
        })?;

        for i in 0..region.len() {
            let pos = start + i as u32;
            let (reported_pos, value) = view
                .next()
                .ok_or_else(|| {
                    ReaderError::Other(anyhow::anyhow!(
                        "unexpected end of d4 view at {}:{} position {pos}",
                        self.shared.path.display(),
                        contig.name,
                    ))
                })?
                .map_err(|e| {
                    ReaderError::Other(anyhow::anyhow!(
                        "failed to read d4 value at {}:{}:{pos}: {e}",
                        self.shared.path.display(),
                        contig.name,
                    ))
                })?;

            if reported_pos != pos {
                return Err(ReaderError::Other(anyhow::anyhow!(
                    "d4 position mismatch at {}:{}:{pos}: got {reported_pos}",
                    self.shared.path.display(),
                    contig.name,
                )));
            }

            let depth = u32::try_from(value).map_err(|_| {
                ReaderError::Other(anyhow::anyhow!(
                    "d4 depth at {}:{}:{pos} cannot be represented as u32 (got {value})",
                    self.shared.path.display(),
                    contig.name,
                ))
            })?;

            dst[[i, 0]] = depth;
        }

        Ok(())
    }
    fn fork(&self) -> Result<Self> {
        let file = File::open(&self.shared.path).map_err(|source| ReaderError::Io {
            path: self.shared.path.clone(),
            source,
        })?;
        let reader = D4TrackReader::from_reader(file, None).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "failed to fork d4 track for {}: {e}",
                self.shared.path.display(),
            ))
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            inner: Mutex::new(reader),
        })
    }
}
