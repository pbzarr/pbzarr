use color_eyre::Result;
use color_eyre::eyre::eyre;
use ndarray::{Array2, ArrayView1};
use pbzarr::{PbzStore, Track, parse_region};
use std::io::Write;
use std::ops::Range;

use crate::io::Format;
use crate::io::bed_writer::BedWriter;
use crate::io::bedgraph_writer::BedGraphWriter;
use crate::io::tsv_writer::TsvWriter;

pub struct ContigSlice {
    pub name: String,
    pub length: u64,
    pub range: Range<u64>,
}

pub struct ExportPlan {
    pub contigs: Vec<ContigSlice>,
    pub column_indices: Vec<usize>,
}

/// Build an export plan from CLI arguments.
///
/// Resolves `--region` (None = all contigs in store order, full length each).
/// Resolves `--column` (empty = all columns; otherwise validate and produce
/// an ordered list of indices into the track's columns).
pub fn build_plan(
    store: &PbzStore,
    track: &Track,
    region: Option<&str>,
    columns: &[String],
) -> Result<ExportPlan> {
    let contigs = match region {
        Some(s) => {
            let r = parse_region(s)?;
            let length = store.contig_length(&r.contig)?;
            let start = r.start.unwrap_or(0);
            let end = r.end.unwrap_or(length).min(length);
            if start >= end {
                return Err(eyre!("region '{s}' is empty (start={start} >= end={end})"));
            }
            vec![ContigSlice {
                name: r.contig,
                length,
                range: start..end,
            }]
        }
        None => {
            let mut out = Vec::new();
            for c in store.contigs() {
                let length = store.contig_length(c)?;
                if length == 0 {
                    continue;
                }
                out.push(ContigSlice {
                    name: c.clone(),
                    length,
                    range: 0..length,
                });
            }
            out
        }
    };

    let column_indices = if columns.is_empty() {
        match track.columns() {
            Some(cs) => (0..cs.len()).collect(),
            None => Vec::new(),
        }
    } else {
        let track_cols = track.columns().ok_or_else(|| {
            eyre!(
                "track '{}' has no columns; --column cannot be used",
                track.name()
            )
        })?;
        let mut out = Vec::with_capacity(columns.len());
        for name in columns {
            let idx = track_cols.iter().position(|c| c == name).ok_or_else(|| {
                pbzarr::PbzError::ColumnNotFound {
                    name: name.clone(),
                    available: Some(track_cols.to_vec()),
                }
            })?;
            out.push(idx);
        }
        out
    };

    Ok(ExportPlan {
        contigs,
        column_indices,
    })
}

pub enum WriterChoice<W: Write> {
    Tsv(TsvWriter<W>),
    BedGraphU8(BedGraphWriter<W, u8>),
    BedGraphU16(BedGraphWriter<W, u16>),
    BedGraphU32(BedGraphWriter<W, u32>),
    BedGraphI8(BedGraphWriter<W, i8>),
    BedGraphI16(BedGraphWriter<W, i16>),
    BedGraphI32(BedGraphWriter<W, i32>),
    BedGraphF32(BedGraphWriter<W, f32>),
    BedGraphF64(BedGraphWriter<W, f64>),
    Bed(BedWriter<W>),
}

pub fn make_writer<W: Write>(
    fmt: Format,
    w: W,
    track: &Track,
    include_zero: bool,
) -> Result<WriterChoice<W>> {
    match (fmt, track.dtype()) {
        (Format::Tsv, _) => Ok(WriterChoice::Tsv(TsvWriter::new(w))),
        (Format::Bed, "bool") => Ok(WriterChoice::Bed(BedWriter::new(w))),
        (Format::Bed, _) => Err(eyre!("BED requires dtype=bool")),
        (Format::Bedgraph, "uint8") => Ok(WriterChoice::BedGraphU8(BedGraphWriter::new(
            w,
            0u8,
            include_zero,
        ))),
        (Format::Bedgraph, "uint16") => Ok(WriterChoice::BedGraphU16(BedGraphWriter::new(
            w,
            0u16,
            include_zero,
        ))),
        (Format::Bedgraph, "uint32") => Ok(WriterChoice::BedGraphU32(BedGraphWriter::new(
            w,
            0u32,
            include_zero,
        ))),
        (Format::Bedgraph, "int8") => Ok(WriterChoice::BedGraphI8(BedGraphWriter::new(
            w,
            0i8,
            include_zero,
        ))),
        (Format::Bedgraph, "int16") => Ok(WriterChoice::BedGraphI16(BedGraphWriter::new(
            w,
            0i16,
            include_zero,
        ))),
        (Format::Bedgraph, "int32") => Ok(WriterChoice::BedGraphI32(BedGraphWriter::new(
            w,
            0i32,
            include_zero,
        ))),
        (Format::Bedgraph, "float32") => Ok(WriterChoice::BedGraphF32(BedGraphWriter::new(
            w,
            0f32,
            include_zero,
        ))),
        (Format::Bedgraph, "float64") => Ok(WriterChoice::BedGraphF64(BedGraphWriter::new(
            w,
            0f64,
            include_zero,
        ))),
        (Format::Bedgraph, d) => Err(eyre!("unsupported dtype for bedgraph: {d}")),
    }
}

pub fn run_export<W: Write>(
    track: &Track,
    plan: &ExportPlan,
    mut writer: WriterChoice<W>,
) -> Result<()> {
    // Emit TSV header up-front.
    if let WriterChoice::Tsv(ref mut w) = writer {
        if track.has_columns() {
            let cols = track.columns().unwrap_or(&[]);
            let selected: Vec<&str> = plan
                .column_indices
                .iter()
                .filter_map(|&i| cols.get(i).map(|s| s.as_str()))
                .collect();
            w.header(&selected)?;
        } else {
            w.header_scalar()?;
        }
    }

    for slice in &plan.contigs {
        if slice.range.start >= slice.range.end {
            continue;
        }
        let chunk_range = track.overlapping_chunks(slice.range.start, slice.range.end);
        for chunk_idx in chunk_range {
            let (cs, ce) = track.chunk_bounds(chunk_idx, slice.length);
            let pos_start = slice.range.start.max(cs);
            let pos_end = slice.range.end.min(ce);
            if pos_start >= pos_end {
                continue;
            }
            let pos_slice = pos_start..pos_end;
            dispatch(
                track,
                &slice.name,
                chunk_idx,
                cs,
                pos_slice,
                plan,
                &mut writer,
            )?;
        }
    }

    match writer {
        WriterChoice::Tsv(mut w) => w.finish()?,
        WriterChoice::BedGraphU8(mut w) => w.finish()?,
        WriterChoice::BedGraphU16(mut w) => w.finish()?,
        WriterChoice::BedGraphU32(mut w) => w.finish()?,
        WriterChoice::BedGraphI8(mut w) => w.finish()?,
        WriterChoice::BedGraphI16(mut w) => w.finish()?,
        WriterChoice::BedGraphI32(mut w) => w.finish()?,
        WriterChoice::BedGraphF32(mut w) => w.finish()?,
        WriterChoice::BedGraphF64(mut w) => w.finish()?,
        WriterChoice::Bed(mut w) => w.finish()?,
    }
    Ok(())
}

fn dispatch<W: Write>(
    track: &Track,
    contig: &str,
    chunk_idx: u64,
    cs: u64,
    pos_slice: Range<u64>,
    plan: &ExportPlan,
    writer: &mut WriterChoice<W>,
) -> Result<()> {
    match track.dtype() {
        "uint8" => emit_numeric_u8(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "uint16" => emit_numeric_u16(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "uint32" => emit_numeric_u32(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "int8" => emit_numeric_i8(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "int16" => emit_numeric_i16(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "int32" => emit_numeric_i32(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "float32" => emit_numeric_f32(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "float64" => emit_numeric_f64(track, contig, chunk_idx, cs, pos_slice, plan, writer),
        "bool" => emit_bool(track, contig, chunk_idx, cs, pos_slice, writer),
        d => Err(eyre!("unsupported track dtype: {d}")),
    }
}

/// For columnar tracks with one column (or a 2D chunk we want a single column
/// from), extract column 0 (or `col_indices[0]`) as a contiguous owned 1D
/// array. Used by bedGraph which only sees a single column at a time.
fn pick_single_column<T: Clone>(chunk: &Array2<T>, col_indices: &[usize]) -> ndarray::Array1<T> {
    let col = col_indices.first().copied().unwrap_or(0);
    chunk.column(col).to_owned()
}

macro_rules! emit_numeric {
    ($fname:ident, $ty:ty, $tsv_2d:ident, $tsv_1d:ident, $bg_variant:ident, $bg_emit:ident) => {
        fn $fname<W: Write>(
            track: &Track,
            contig: &str,
            chunk_idx: u64,
            cs: u64,
            pos_slice: Range<u64>,
            plan: &ExportPlan,
            writer: &mut WriterChoice<W>,
        ) -> Result<()> {
            if track.has_columns() {
                let chunk: Array2<$ty> = track.read_chunk::<$ty>(contig, chunk_idx)?;
                match writer {
                    WriterChoice::Tsv(w) => {
                        w.$tsv_2d(contig, cs, pos_slice, chunk.view(), &plan.column_indices)?;
                    }
                    WriterChoice::$bg_variant(w) => {
                        let col = pick_single_column(&chunk, &plan.column_indices);
                        let view: ArrayView1<$ty> = col.view();
                        w.$bg_emit(contig, cs, pos_slice, view)?;
                    }
                    _ => {
                        return Err(eyre!(
                            "internal error: writer/dtype mismatch for {}",
                            track.dtype()
                        ));
                    }
                }
            } else {
                let chunk: ndarray::Array1<$ty> = track.read_chunk_1d::<$ty>(contig, chunk_idx)?;
                match writer {
                    WriterChoice::Tsv(w) => {
                        w.$tsv_1d(contig, cs, pos_slice, chunk.view())?;
                    }
                    WriterChoice::$bg_variant(w) => {
                        w.$bg_emit(contig, cs, pos_slice, chunk.view())?;
                    }
                    _ => {
                        return Err(eyre!(
                            "internal error: writer/dtype mismatch for {}",
                            track.dtype()
                        ));
                    }
                }
            }
            Ok(())
        }
    };
}

emit_numeric!(
    emit_numeric_u8,
    u8,
    emit_u8,
    emit_u8_scalar,
    BedGraphU8,
    emit_u8
);
emit_numeric!(
    emit_numeric_u16,
    u16,
    emit_u16,
    emit_u16_scalar,
    BedGraphU16,
    emit_u16
);
emit_numeric!(
    emit_numeric_u32,
    u32,
    emit_u32,
    emit_u32_scalar,
    BedGraphU32,
    emit_u32
);
emit_numeric!(
    emit_numeric_i8,
    i8,
    emit_i8,
    emit_i8_scalar,
    BedGraphI8,
    emit_i8
);
emit_numeric!(
    emit_numeric_i16,
    i16,
    emit_i16,
    emit_i16_scalar,
    BedGraphI16,
    emit_i16
);
emit_numeric!(
    emit_numeric_i32,
    i32,
    emit_i32,
    emit_i32_scalar,
    BedGraphI32,
    emit_i32
);
emit_numeric!(
    emit_numeric_f32,
    f32,
    emit_f32,
    emit_f32_scalar,
    BedGraphF32,
    emit_f32
);
emit_numeric!(
    emit_numeric_f64,
    f64,
    emit_f64,
    emit_f64_scalar,
    BedGraphF64,
    emit_f64
);

fn emit_bool<W: Write>(
    track: &Track,
    contig: &str,
    chunk_idx: u64,
    cs: u64,
    pos_slice: Range<u64>,
    writer: &mut WriterChoice<W>,
) -> Result<()> {
    if track.has_columns() {
        return Err(eyre!(
            "columnar bool tracks are not supported by export/cat (track '{}')",
            track.name()
        ));
    }
    let chunk: ndarray::Array1<bool> = track.read_chunk_1d::<bool>(contig, chunk_idx)?;
    match writer {
        WriterChoice::Tsv(w) => {
            w.emit_bool_scalar(contig, cs, pos_slice, chunk.view())?;
        }
        WriterChoice::Bed(w) => {
            w.emit(contig, cs, pos_slice, chunk.view())?;
        }
        _ => {
            return Err(eyre!(
                "internal error: writer/dtype mismatch for bool track"
            ));
        }
    }
    Ok(())
}
