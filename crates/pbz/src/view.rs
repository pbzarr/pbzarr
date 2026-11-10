//! `pbz view`: stream one track as bedGraph-like text.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail};
use noodles_bgzf as bgzf;
use pbzarr::io::{Dtype, Numeric};
use pbzarr::{Dataset, Measurement, RegionQuery};
use rayon::prelude::*;

use crate::regions::{NamedRegion, select_regions};

/// Target bytes fetched per batch over the selected columns.
const BATCH_TARGET_BYTES: u64 = 64 << 20;
const BATCH_MAX_ROWS: u64 = 4 << 20;

pub(crate) struct ViewSpec {
    pub store: PathBuf,
    pub track: Option<String>,
    pub regions: Vec<RegionQuery>,
    pub regions_file: Option<PathBuf>,
    pub columns: Vec<String>,
    pub output: Option<PathBuf>,
    pub no_header: bool,
    pub threads: Option<NonZeroUsize>,
    pub precision: u8,
}

pub(crate) fn run_view(spec: &ViewSpec) -> Result<()> {
    if !(1..=17).contains(&spec.precision) {
        bail!("--precision must be between 1 and 17 significant digits");
    }
    // One rayon pool serves both zarrs chunk decode and BGZF compression.
    if let Some(threads) = spec.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .build_global()
            .context("configure worker thread pool")?;
    }
    let store = open_store(&spec.store)?;
    let measurement = resolve_measurement(&store, spec.track.as_deref())?;
    let (columns, names) = resolve_columns(&measurement, &spec.columns)?;
    let regions = select_regions(
        measurement.reference(),
        &spec.regions,
        spec.regions_file.as_ref(),
    )?;
    let precision = usize::from(spec.precision);
    match spec.output.as_deref() {
        None => {
            let mut out = BufWriter::new(io::stdout());
            stream(
                &measurement,
                &regions,
                &columns,
                &names,
                spec.no_header,
                precision,
                &mut out,
            )?;
            out.flush()?;
        }
        Some(path)
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gz")) =>
        {
            let file =
                File::create(path).with_context(|| format!("create output {}", path.display()))?;
            // Deflate dominates the export cost; compress blocks on a pool.
            let mut out = bgzf::io::MultithreadedWriter::new(file);
            stream(
                &measurement,
                &regions,
                &columns,
                &names,
                spec.no_header,
                precision,
                &mut out,
            )?;
            out.finish()?;
        }
        Some(path) => {
            let file =
                File::create(path).with_context(|| format!("create output {}", path.display()))?;
            let mut out = BufWriter::new(file);
            stream(
                &measurement,
                &regions,
                &columns,
                &names,
                spec.no_header,
                precision,
                &mut out,
            )?;
            out.flush()?;
        }
    }
    Ok(())
}

pub(crate) fn open_store(path: &Path) -> Result<Dataset> {
    require_store(path)?;
    Dataset::open_read_only(path).with_context(|| format!("open store {}", path.display()))
}

pub(crate) fn require_store(path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("store {} does not exist", path.display());
    }
    Ok(())
}

/// Track names in sorted order.
fn track_names(store: &Dataset) -> Vec<&str> {
    let mut names: Vec<&str> = store
        .variables()
        .values()
        .filter(|v| v.is_track())
        .map(|v| v.name.as_str())
        .collect();
    names.sort_unstable();
    names
}

pub(crate) fn resolve_measurement<'s>(
    store: &'s Dataset,
    requested: Option<&str>,
) -> Result<Measurement<'s>> {
    let available = track_names(store);
    match requested {
        Some(name) => {
            if !available.contains(&name) {
                bail!(
                    "track {name:?} is not in the store (available: {})",
                    available.join(", ")
                );
            }
            Ok(Measurement::root(store, name)?)
        }
        None => match available.as_slice() {
            [] => bail!("store has no tracks"),
            [only] => Ok(Measurement::root(store, only)?),
            _ => bail!(
                "store has {} tracks; name one with --track (available: {})",
                available.len(),
                available.join(", ")
            ),
        },
    }
}

/// Column indices to emit (empty for rank-1) and the header names.
fn resolve_columns(
    measurement: &Measurement,
    requested: &[String],
) -> Result<(Vec<usize>, Vec<String>)> {
    if measurement.rank() > 2 {
        bail!(
            "track {:?} with dimensions ({}): view renders a position axis and at most one more",
            measurement.name(),
            measurement.dimensions().join(", ")
        );
    }
    if measurement.rank() == 1 {
        if !requested.is_empty() {
            bail!(
                "--columns was given but track {:?} has no column axis",
                measurement.name()
            );
        }
        return Ok((Vec::new(), vec![measurement.name().to_owned()]));
    }
    let labels = measurement.column_labels()?;
    if requested.is_empty() {
        return Ok(((0..labels.len()).collect(), labels));
    }
    let mut indices = Vec::with_capacity(requested.len());
    for label in requested {
        let index = labels
            .iter()
            .position(|candidate| candidate == label)
            .with_context(|| {
                format!(
                    "column {label:?} is not a column of track {:?}",
                    measurement.name()
                )
            })?;
        indices.push(index);
    }
    Ok((indices, requested.to_vec()))
}

fn stream<W: Write>(
    measurement: &Measurement,
    regions: &[NamedRegion],
    columns: &[usize],
    names: &[String],
    no_header: bool,
    precision: usize,
    out: &mut W,
) -> Result<()> {
    if !no_header {
        write!(out, "#chrom\tstart\tend")?;
        for name in names {
            write!(out, "\t{name}")?;
        }
        writeln!(out)?;
    }
    match measurement.dtype() {
        Dtype::U8 => export::<u8, _>(measurement, regions, columns, precision, out),
        Dtype::U16 => export::<u16, _>(measurement, regions, columns, precision, out),
        Dtype::U32 => export::<u32, _>(measurement, regions, columns, precision, out),
        Dtype::U64 => export::<u64, _>(measurement, regions, columns, precision, out),
        Dtype::I8 => export::<i8, _>(measurement, regions, columns, precision, out),
        Dtype::I16 => export::<i16, _>(measurement, regions, columns, precision, out),
        Dtype::I32 => export::<i32, _>(measurement, regions, columns, precision, out),
        Dtype::I64 => export::<i64, _>(measurement, regions, columns, precision, out),
        Dtype::F32 => export::<f32, _>(measurement, regions, columns, precision, out),
        Dtype::F64 => export::<f64, _>(measurement, regions, columns, precision, out),
        Dtype::Bool => export::<bool, _>(measurement, regions, columns, precision, out),
    }
}

/// Per-dtype cell behavior: run equality, missing-ness, rounding, rendering.
trait Cell: Numeric {
    fn same(self, other: Self) -> bool;
    fn is_missing(self) -> bool {
        false
    }
    /// Round to `sig` significant digits before run comparison, so values
    /// that print identically collapse into one row. Identity for non-floats.
    fn quantize(self, sig: usize) -> Self {
        let _ = sig;
        self
    }
    fn write_to(self, out: &mut impl Write, sig: usize) -> io::Result<()>;
}

macro_rules! impl_cell_int {
    ($($ty:ty),*) => {
        $(impl Cell for $ty {
            fn same(self, other: Self) -> bool {
                self == other
            }
            fn write_to(self, out: &mut impl Write, _sig: usize) -> io::Result<()> {
                write!(out, "{self}")
            }
        })*
    };
}

impl_cell_int!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! impl_cell_float {
    ($($ty:ty),*) => {
        $(impl Cell for $ty {
            /// Bitwise so NaN cells extend runs instead of breaking them.
            fn same(self, other: Self) -> bool {
                self.to_bits() == other.to_bits()
            }
            fn is_missing(self) -> bool {
                self.is_nan()
            }
            fn quantize(self, sig: usize) -> Self {
                if self == 0.0 || !self.is_finite() {
                    return self;
                }
                let exp = f64::from(self).abs().log10().floor() as i32;
                let factor = 10f64.powi(sig as i32 - 1 - exp);
                if !factor.is_finite() || factor == 0.0 {
                    return self;
                }
                ((f64::from(self) * factor).round() / factor) as Self
            }
            fn write_to(self, out: &mut impl Write, sig: usize) -> io::Result<()> {
                crate::fmt::write_g(out, f64::from(self), sig)
            }
        })*
    };
}

impl_cell_float!(f32, f64);

impl Cell for bool {
    fn same(self, other: Self) -> bool {
        self == other
    }
    fn write_to(self, out: &mut impl Write, _sig: usize) -> io::Result<()> {
        write!(out, "{}", u8::from(self))
    }
}

/// One read of a contiguous column range and the output slots it fills.
struct ColumnRead {
    cols: std::ops::Range<u64>,
    /// (output slot, track column) pairs.
    picks: Vec<(usize, usize)>,
}

/// Reads that touch only the inner column chunks holding selected columns.
/// Touched chunks that are next to each other share one read.
fn column_reads(columns: &[usize], inner_col: u64) -> Vec<ColumnRead> {
    let inner_col = inner_col.max(1);
    let mut picks: Vec<(usize, usize)> = columns.iter().copied().enumerate().collect();
    picks.sort_by_key(|&(slot, col)| (col, slot));
    let mut reads: Vec<ColumnRead> = Vec::new();
    for (slot, col) in picks {
        let chunk = col as u64 / inner_col;
        match reads.last_mut() {
            Some(read) if chunk <= (read.cols.end - 1) / inner_col + 1 => {
                read.cols.end = read.cols.end.max(col as u64 + 1);
                read.picks.push((slot, col));
            }
            _ => reads.push(ColumnRead {
                cols: col as u64..col as u64 + 1,
                picks: vec![(slot, col)],
            }),
        }
    }
    reads
}

/// Batch plan on the inner chunk grid of the selected columns.
struct Plan {
    inner_pos: u64,
    batch_chunks: u64,
    /// Empty for rank-1 tracks.
    reads: Vec<ColumnRead>,
    n_out: usize,
}

impl Plan {
    fn new(measurement: &Measurement, columns: &[usize]) -> Result<Self> {
        let (inner_pos, inner_col) = measurement.inner_chunk_shape()?;
        let inner_pos = inner_pos.max(1);
        let reads = if measurement.rank() == 1 {
            Vec::new()
        } else {
            column_reads(columns, inner_col)
        };
        let width: u64 = if reads.is_empty() {
            1
        } else {
            reads
                .iter()
                .map(|read| read.cols.end - read.cols.start)
                .sum()
        };
        let per_chunk = inner_pos * width * measurement.dtype().size() as u64;
        let batch_chunks = (BATCH_TARGET_BYTES / per_chunk.max(1))
            .min(BATCH_MAX_ROWS / inner_pos)
            .max(1);
        Ok(Self {
            inner_pos,
            batch_chunks,
            reads,
            n_out: columns.len().max(1),
        })
    }

    /// End of the batch that starts at flat position `lo`, at most `hi`.
    fn batch_end(&self, lo: u64, hi: u64) -> u64 {
        let first = lo / self.inner_pos * self.inner_pos;
        (first + self.batch_chunks * self.inner_pos).min(hi)
    }

    /// Read one batch as per-output-column vectors, quantized to the output
    /// precision. `read_flat_columns` returns a row-major array, so
    /// `raw[row * width + col]` addresses `(row, col)`.
    fn read<T: Cell>(
        &self,
        measurement: &Measurement,
        positions: std::ops::Range<u64>,
        precision: usize,
    ) -> Result<Vec<Vec<T>>> {
        if self.reads.is_empty() {
            let raw = measurement
                .read_flat::<T>(positions)?
                .into_raw_vec_and_offset()
                .0;
            return Ok(vec![
                raw.into_iter().map(|v| v.quantize(precision)).collect(),
            ]);
        }
        let parts: Vec<Vec<(usize, Vec<T>)>> = self
            .reads
            .par_iter()
            .map(|read| {
                let raw = measurement
                    .read_flat_columns::<T>(positions.clone(), read.cols.clone())?
                    .into_raw_vec_and_offset()
                    .0;
                let width = (read.cols.end - read.cols.start) as usize;
                let first = read.cols.start as usize;
                let rows = raw.len() / width;
                Ok(read
                    .picks
                    .iter()
                    .map(|&(slot, col)| {
                        let values = (0..rows)
                            .map(|row| raw[row * width + col - first].quantize(precision))
                            .collect();
                        (slot, values)
                    })
                    .collect())
            })
            .collect::<Result<_>>()?;
        let mut out: Vec<Vec<T>> = (0..self.n_out).map(|_| Vec::new()).collect();
        for (slot, values) in parts.into_iter().flatten() {
            out[slot] = values;
        }
        Ok(out)
    }
}

/// Stream each region in order, collapsing equal-value runs. Runs do not
/// cross region edges.
fn export<T: Cell, W: Write>(
    measurement: &Measurement,
    regions: &[NamedRegion],
    columns: &[usize],
    precision: usize,
    out: &mut W,
) -> Result<()> {
    let plan = Plan::new(measurement, columns)?;
    let offsets = measurement.reference().offsets();
    for named in regions {
        let base = offsets[named.region.contig.as_usize()];
        let chrom = named.chrom.as_str();
        // Open run in this region: (start, end, values).
        let mut run: Option<(u64, u64, Vec<T>)> = None;
        let mut start = named.region.start;
        while start < named.region.end {
            let end = plan.batch_end(base + start, base + named.region.end) - base;
            let batch = plan.read::<T>(measurement, base + start..base + end, precision)?;
            let rows = batch.first().map_or(0, Vec::len);
            for row in 0..rows {
                let pos = start + row as u64;
                if batch.iter().all(|column| column[row].is_missing()) {
                    flush(out, chrom, &mut run, precision)?;
                    continue;
                }
                let extends = run.as_ref().is_some_and(|(_, run_end, values)| {
                    *run_end == pos
                        && batch
                            .iter()
                            .zip(values)
                            .all(|(column, value)| column[row].same(*value))
                });
                if extends {
                    run.as_mut().expect("checked above").1 += 1;
                    continue;
                }
                flush(out, chrom, &mut run, precision)?;
                run = Some((
                    pos,
                    pos + 1,
                    batch.iter().map(|column| column[row]).collect(),
                ));
            }
            start = end;
        }
        flush(out, chrom, &mut run, precision)?;
    }
    Ok(())
}

fn flush<T: Cell>(
    out: &mut impl Write,
    contig: &str,
    run: &mut Option<(u64, u64, Vec<T>)>,
    precision: usize,
) -> io::Result<()> {
    let Some((start, end, values)) = run.take() else {
        return Ok(());
    };
    write!(out, "{contig}\t{start}\t{end}")?;
    for value in values {
        out.write_all(b"\t")?;
        value.write_to(&mut *out, precision)?;
    }
    out.write_all(b"\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_reads_skip_untouched_chunks_and_merge_neighbors() {
        let reads = column_reads(&[9, 0, 3], 2);
        let spans: Vec<_> = reads.iter().map(|r| r.cols.clone()).collect();
        assert_eq!(spans, vec![0..4, 9..10]);
        assert_eq!(reads[0].picks, vec![(1, 0), (2, 3)]);
    }

    #[test]
    fn quantize_collapses_values_that_print_identically() {
        let a = 0.5000001f32.quantize(6);
        let b = 0.5f32.quantize(6);
        assert!(a.same(b));
    }
}
