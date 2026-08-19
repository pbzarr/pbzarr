//! D4 single-track reader implementing `ValueReader`.
//!
//! Uses the mmap-backed `d4::D4TrackReader` (default generics
//! `<BitArrayReader, SparseArrayReader<RangeRecord>>`) and reads each region
//! via `.split(Some(chunk_size))` + `ptab.to_codec().decode_block(...)`. The
//! reference for this pattern is `pyd4::load_values_to_buffer` in the upstream
//! d4 repo.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use d4::D4TrackReader;
use d4::ptab::{DecodeResult, Decoder};
use d4::stab::SecondaryTablePartReader;

use pbzarr::genome::{Contig, Genome};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ReaderError, Result, ValueReader};

/// Read a d4 file's contig list from its header without opening the data path.
///
/// Returns `(name, length)` pairs in file order, so a store can be sized from
/// the source without pyd4 or an external d4tools invocation.
pub fn contigs<P: AsRef<Path>>(src: P) -> Result<Vec<(String, u64)>> {
    let path = src.as_ref();
    // Pin the default table generics, as the `D4Reader::inner` field does;
    // `open_first_track` is otherwise ambiguous over its type parameters.
    let reader: D4TrackReader =
        D4TrackReader::open_first_track(path).map_err(|source| ReaderError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(reader
        .header()
        .chrom_list()
        .iter()
        .map(|c| (c.name.clone(), c.size as u64))
        .collect())
}

struct Shared {
    path: PathBuf,
    genome: Genome,
    schema: OutputSchema,
}

/// D4 single-sample reader. Owns one mmap-backed `D4TrackReader`; multi-thread
/// callers should use `fork()` to get a per-thread handle on the same file.
pub struct D4Reader {
    shared: Arc<Shared>,
    inner: D4TrackReader,
}

impl D4Reader {
    /// Open a d4 file, taking its first data track. Reads the header to build
    /// the canonical `Genome`.
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
            shared: Arc::new(Shared {
                path,
                genome,
                schema: OutputSchema::single("depth", Dtype::I32),
            }),
            inner,
        })
    }

    fn fork_handle(&self) -> Result<Self> {
        let reader = D4TrackReader::open_first_track(&self.shared.path).map_err(|source| {
            ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            }
        })?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            inner: reader,
        })
    }
}

/// `emit` receives each covered base as an offset from `start`.
fn decode_region(
    shared: &Shared,
    inner: &mut D4TrackReader,
    contig_name: &str,
    start: u64,
    end: u64,
    mut emit: impl FnMut(usize, i32),
) -> Result<()> {
    if end <= start {
        return Ok(());
    }

    // Resolve the name in the d4 file's own genome; the caller's
    // ContigId (if it had one) belongs to a different namespace.
    if shared.genome.id(contig_name).is_none() {
        return Err(ReaderError::ContigNotFound {
            path: shared.path.clone(),
            contig: contig_name.to_owned(),
        });
    }
    // d4's region API is natively 0-based half-open; no conversion needed,
    // only the narrowing cast to its u32 coordinate type.
    let start_u32 = u32::try_from(start).map_err(|_| {
        ReaderError::Other(anyhow::anyhow!(
            "d4 read requires start <= u32::MAX, got {} in {}",
            start,
            shared.path.display(),
        ))
    })?;
    let end_u32 = u32::try_from(end).map_err(|_| {
        ReaderError::Other(anyhow::anyhow!(
            "d4 read requires end <= u32::MAX, got {} in {}",
            end,
            shared.path.display(),
        ))
    })?;
    let span = (end_u32 - start_u32) as usize;

    // split() requires &mut self; the partitioning is cheap relative to
    // the actual decode of a 1 Mbp range. Sizing the limit at the request
    // span gives at most one partition per contig overlapping our range.
    let partitions = inner
        .split(Some(span.max(1)))
        .map_err(|source| ReaderError::Io {
            path: shared.path.clone(),
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
            emit(pos - start_u32 as usize, resolved);
        });
    }

    Ok(())
}

impl ValueReader for D4Reader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    // d4's native dtype is i32 (signed; some signal tracks emit negatives).
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
                    "d4 reader produces 1 field but the engine handed {} sink(s)",
                    outputs.len()
                ),
            });
        };
        let dst = output.as_i32_mut()?;
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
