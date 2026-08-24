//! `pbz view`: stream one track as bedGraph-like text.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use noodles_bgzf as bgzf;
use pbzarr::io::{Dtype, Numeric};
use pbzarr::{PbzStore, Region, Track};

/// Target bytes fetched per batch (full stored width).
const BATCH_TARGET_BYTES: usize = 64 << 20;

pub(crate) struct ViewSpec {
    pub store: PathBuf,
    pub track: Option<String>,
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
    let store = PbzStore::open(&spec.store)
        .with_context(|| format!("open store {}", spec.store.display()))?;
    let track = resolve_track(&store, spec.track.as_deref())?;
    let (columns, names) = resolve_columns(track, &spec.columns)?;
    let precision = usize::from(spec.precision);
    match spec.output.as_deref() {
        None => {
            let mut out = BufWriter::new(io::stdout());
            stream(track, &columns, &names, spec.no_header, precision, &mut out)?;
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
            stream(track, &columns, &names, spec.no_header, precision, &mut out)?;
            out.finish()?;
        }
        Some(path) => {
            let file =
                File::create(path).with_context(|| format!("create output {}", path.display()))?;
            let mut out = BufWriter::new(file);
            stream(track, &columns, &names, spec.no_header, precision, &mut out)?;
            out.flush()?;
        }
    }
    Ok(())
}

fn resolve_track<'s>(store: &'s PbzStore, requested: Option<&str>) -> Result<&'s Track> {
    let mut available: Vec<&str> = store.track_names().collect();
    available.sort_unstable();
    match requested {
        Some(name) => store.track(name).with_context(|| {
            format!(
                "track {name:?} is not in the store (available: {})",
                available.join(", ")
            )
        }),
        None => match available.as_slice() {
            [] => bail!("store has no tracks"),
            [only] => Ok(store.track(only).expect("listed track exists")),
            _ => bail!(
                "store has {} tracks; name one (available: {})",
                available.len(),
                available.join(", ")
            ),
        },
    }
}

/// Column indices to emit (empty for rank-1) and the header names.
fn resolve_columns(track: &Track, requested: &[String]) -> Result<(Vec<usize>, Vec<String>)> {
    if track.rank() == 1 {
        if !requested.is_empty() {
            bail!(
                "--columns was given but track {:?} has no column axis",
                track.name()
            );
        }
        return Ok((Vec::new(), vec![track.name().to_owned()]));
    }
    let labels = track.column_labels()?;
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
                    track.name()
                )
            })?;
        indices.push(index);
    }
    Ok((indices, requested.to_vec()))
}

fn stream<W: Write>(
    track: &Track,
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
    match track.dtype() {
        Dtype::U8 => export::<u8, _>(track, columns, precision, out),
        Dtype::U16 => export::<u16, _>(track, columns, precision, out),
        Dtype::U32 => export::<u32, _>(track, columns, precision, out),
        Dtype::I8 => export::<i8, _>(track, columns, precision, out),
        Dtype::I16 => export::<i16, _>(track, columns, precision, out),
        Dtype::I32 => export::<i32, _>(track, columns, precision, out),
        Dtype::F32 => export::<f32, _>(track, columns, precision, out),
        Dtype::F64 => export::<f64, _>(track, columns, precision, out),
        Dtype::Bool => export::<bool, _>(track, columns, precision, out),
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

impl_cell_int!(u8, u16, u32, i8, i16, i32);

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

fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

/// Read one batch as per-output-column vectors, quantized to the output
/// precision. `columns` is empty for rank-1 tracks. `read_region` returns a
/// standard-layout (row-major) array, so `raw[row * n_cols + col]` addresses
/// `(row, col)`.
fn read_columns<T: Cell>(
    track: &Track,
    region: &Region,
    columns: &[usize],
    precision: usize,
) -> Result<Vec<Vec<T>>> {
    let array = track.read_region::<T>(region)?;
    if track.rank() == 1 {
        let raw = array.into_raw_vec_and_offset().0;
        return Ok(vec![
            raw.into_iter().map(|v| v.quantize(precision)).collect(),
        ]);
    }
    let n_cols = array.shape()[1];
    let raw = array.into_raw_vec_and_offset().0;
    let rows = raw.len() / n_cols.max(1);
    Ok(columns
        .iter()
        .map(|&col| {
            (0..rows)
                .map(|row| raw[row * n_cols + col].quantize(precision))
                .collect()
        })
        .collect())
}

/// Stream the whole track in genome order, collapsing equal-value runs.
/// One `read_region` per batch; zarrs decodes its chunks concurrently.
fn export<T: Cell, W: Write>(
    track: &Track,
    columns: &[usize],
    precision: usize,
    out: &mut W,
) -> Result<()> {
    let per_row = track.columns_count()? * dtype_size(track.dtype());
    let batch_rows = (BATCH_TARGET_BYTES / per_row.max(1)).clamp(1, 4 << 20) as u64;
    for (id, contig) in track.genome().iter() {
        // Open run on this contig: (start, end, values).
        let mut run: Option<(u64, u64, Vec<T>)> = None;
        let mut start = 0u64;
        while start < contig.length {
            let end = (start + batch_rows).min(contig.length);
            let region = Region {
                contig: id,
                start,
                end,
            };
            let batch = read_columns::<T>(track, &region, columns, precision)?;
            let rows = batch.first().map_or(0, Vec::len);
            for row in 0..rows {
                let pos = start + row as u64;
                if batch.iter().all(|column| column[row].is_missing()) {
                    flush(out, &contig.name, &mut run, precision)?;
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
                flush(out, &contig.name, &mut run, precision)?;
                run = Some((
                    pos,
                    pos + 1,
                    batch.iter().map(|column| column[row]).collect(),
                ));
            }
            start = end;
        }
        flush(out, &contig.name, &mut run, precision)?;
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
    fn quantize_collapses_values_that_print_identically() {
        let a = 0.5000001f32.quantize(6);
        let b = 0.5f32.quantize(6);
        assert!(a.same(b));
    }
}
