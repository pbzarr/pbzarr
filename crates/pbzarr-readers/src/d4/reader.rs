//! D4 single-track reader implementing `ValueReader`.
//!
//! Uses the mmap-backed `d4::D4TrackReader` (default generics
//! `<BitArrayReader, SparseArrayReader<RangeRecord>>`). `split(None)` is
//! called once per handle to get one `(ptab, stab)` partition per contig;
//! each read then decodes its subrange via
//! `ptab.to_codec().decode_block(...)`. A per-read `split(Some(span))`
//! partitions the whole genome each call, which is quadratic on
//! fragmented assemblies with many small contigs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use d4::D4TrackReader;
use d4::ptab::{BitArrayPartReader, DecodeResult, Decoder};
use d4::stab::{RangeRecord, SecondaryTablePartReader, SparseArrayPartReader};

use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ReaderError, Result, ValueReader};
use pbzarr::{Contig, Reference};

type ContigPart = (BitArrayPartReader, SparseArrayPartReader<RangeRecord>);

struct Shared {
    path: PathBuf,
    genome: Reference,
    schema: OutputSchema,
}

/// D4 single-sample reader. Owns one mmap-backed `D4TrackReader` plus its
/// per-contig partitions; multi-thread callers should use `fork()` to get a
/// per-thread handle on the same file.
pub struct D4Reader {
    shared: Arc<Shared>,
    // Keeps the mmap root alive for the partitions borrowed from it.
    _inner: D4TrackReader,
    parts: HashMap<String, ContigPart>,
}

fn contig_partitions(
    inner: &mut D4TrackReader,
    path: &Path,
) -> Result<HashMap<String, ContigPart>> {
    let partitions = inner.split(None).map_err(|source| ReaderError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut parts = HashMap::with_capacity(partitions.len());
    for (ptab, stab) in partitions {
        let name = ptab.region().0.to_string();
        parts.insert(name, (ptab, stab));
    }
    Ok(parts)
}

impl D4Reader {
    /// Open a d4 file, taking its first data track. The header gives the
    /// contig list.
    pub fn open<P: AsRef<Path>>(src: P) -> Result<Self> {
        let path = src.as_ref().to_path_buf();
        let mut inner =
            D4TrackReader::open_first_track(&path).map_err(|source| ReaderError::Io {
                path: path.clone(),
                source,
            })?;

        let header = inner.header();
        if !header.is_integral() {
            let denominator = header.get_denominator();
            return Err(ReaderError::Other(anyhow::anyhow!(
                "{}: fix-point d4 (denominator {}) is not supported; pbz imports integer d4 files only",
                path.display(),
                denominator,
            )));
        }

        let contigs: Vec<Contig> = header
            .chrom_list()
            .iter()
            .map(|c| Contig {
                name: c.name.clone(),
                length: c.size as u64,
            })
            .collect();
        let genome = Reference::new(contigs).map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "invalid contig list in {}: {e}",
                path.display(),
            ))
        })?;

        let parts = contig_partitions(&mut inner, &path)?;

        Ok(Self {
            shared: Arc::new(Shared {
                path,
                genome,
                schema: OutputSchema::single("depth", Dtype::I32),
            }),
            _inner: inner,
            parts,
        })
    }

    fn fork_handle(&self) -> Result<Self> {
        let mut inner = D4TrackReader::open_first_track(&self.shared.path).map_err(|source| {
            ReaderError::Io {
                path: self.shared.path.clone(),
                source,
            }
        })?;
        let parts = contig_partitions(&mut inner, &self.shared.path)?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            _inner: inner,
            parts,
        })
    }
}

/// `emit` receives each covered base as an offset from `start`.
fn decode_region(
    shared: &Shared,
    parts: &mut HashMap<String, ContigPart>,
    contig_name: &str,
    start: u64,
    end: u64,
    mut emit: impl FnMut(usize, i32),
) -> Result<()> {
    if end <= start {
        return Ok(());
    }

    let Some((ptab, stab)) = parts.get_mut(contig_name) else {
        return Err(ReaderError::ContigNotFound {
            path: shared.path.clone(),
            contig: contig_name.to_owned(),
        });
    };

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

    let (_, part_begin, part_end) = ptab.region();
    let from = start_u32.max(part_begin);
    let to = end_u32.min(part_end);
    if from >= to {
        return Ok(());
    }

    let mut codec = ptab.to_codec();
    codec.decode_block(from as usize, (to - from) as usize, |pos, value| {
        let resolved = match value {
            DecodeResult::Definitely(v) => v,
            DecodeResult::Maybe(v) => stab.decode(pos as u32).unwrap_or(v),
        };
        emit(pos - start_u32 as usize, resolved);
    });

    Ok(())
}

impl ValueReader for D4Reader {
    fn contigs(&self) -> &Reference {
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
            &mut self.parts,
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
