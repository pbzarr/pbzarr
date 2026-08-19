//! bigWig single-track reader implementing `ValueReader`.
//!
//! Backed by `bigtools::BigWigRead` over a `ReopenableFile`. Each region is read
//! with `BigWigRead::values`, which returns a per-base `Vec<f32>` with `NaN` for
//! positions the file does not cover. Those `NaN`s pass through unchanged: a
//! bigWig holds no value at an uncovered base, so the gap is missing data
//! rather than zero.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bigtools::BigWigRead;
use bigtools::utils::file::reopen::ReopenableFile;

use pbzarr::genome::{Contig, Genome};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ReaderError, Result, ValueReader};

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
    schema: OutputSchema,
}

/// bigWig single-sample reader. Owns one file-backed `BigWigRead`; multi-thread
/// callers should use `fork()` to get a per-thread handle on the same file.
pub struct BigWigReader {
    shared: Arc<Shared>,
    inner: BigWigRead<ReopenableFile>,
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
            shared: Arc::new(Shared {
                path,
                genome,
                schema: OutputSchema::single("signal", Dtype::F32),
            }),
            inner,
        })
    }

    fn fork_handle(&self) -> Result<Self> {
        let reader = BigWigRead::open_file(&self.shared.path).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!("open {}: {e}", self.shared.path.display(),))
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            inner: reader,
        })
    }
}

/// `emit` receives each base as an offset from `start`, `NaN` where the file
/// has no coverage.
fn decode_region(
    shared: &Shared,
    inner: &mut BigWigRead<ReopenableFile>,
    contig_name: &str,
    start: u64,
    end: u64,
    mut emit: impl FnMut(usize, f32),
) -> Result<()> {
    if end <= start {
        return Ok(());
    }

    // Resolve the name in the bigWig's own genome; the caller's ContigId
    // (if it had one) belongs to a different namespace.
    if shared.genome.id(contig_name).is_none() {
        return Err(ReaderError::ContigNotFound {
            path: shared.path.clone(),
            contig: contig_name.to_owned(),
        });
    }
    // bigWig's values() is natively 0-based half-open; no conversion
    // needed, only the narrowing cast to its u32 coordinate type.
    let start_u32 = u32::try_from(start).map_err(|_| {
        ReaderError::Other(anyhow::anyhow!(
            "bigWig read requires start <= u32::MAX, got {} in {}",
            start,
            shared.path.display(),
        ))
    })?;
    let end_u32 = u32::try_from(end).map_err(|_| {
        ReaderError::Other(anyhow::anyhow!(
            "bigWig read requires end <= u32::MAX, got {} in {}",
            end,
            shared.path.display(),
        ))
    })?;

    // values() returns a per-base Vec<f32> of length (end - start), with NaN at
    // every position the file does not cover.
    let values = inner.values(contig_name, start_u32, end_u32).map_err(|e| {
        ReaderError::Other(anyhow::anyhow!(
            "read {contig_name}:{start_u32}-{end_u32} in {}: {e}",
            shared.path.display(),
        ))
    })?;

    for (idx, v) in values.into_iter().enumerate() {
        emit(idx, v);
    }

    Ok(())
}

impl ValueReader for BigWigReader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    // bigWig stores a single f32 value per base.
    fn output_schema(&self) -> &OutputSchema {
        &self.shared.schema
    }

    fn read_into(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> Result<()> {
        let [output] = outputs else {
            return Err(ReaderError::SchemaMismatch {
                message: format!(
                    "bigWig reader produces 1 field but the engine handed {} sink(s)",
                    outputs.len()
                ),
            });
        };
        let dst = output.as_f32_mut()?;
        decode_region(
            &self.shared,
            &mut self.inner,
            contig,
            start,
            end,
            |idx, v| {
                dst[idx] = v;
            },
        )
    }

    fn fork(&self) -> Result<Self> {
        self.fork_handle()
    }
}
