//! Dtype-erased column buffers for the multi-column import pipeline. One BED
//! file feeds several tracks of different dtypes in a single decode pass, so
//! the pipeline holds one buffer per track behind a closed enum rather than a
//! single generic `T`.

use std::fmt::Display;
use std::ops::Range;

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, ArrayViewMut2, Axis};

use crate::io::dtype::Dtype;
use crate::io::error::{ReaderError, Result};
use crate::track::Track;

/// One track's owning scratch buffer for a chunk region, tagged by dtype.
pub enum ColumnBuffer {
    U8(Array1<u8>),
    U16(Array1<u16>),
    U32(Array1<u32>),
    I8(Array1<i8>),
    I16(Array1<i16>),
    I32(Array1<i32>),
    F32(Array1<f32>),
    F64(Array1<f64>),
    Bool(Array1<bool>),
}

impl ColumnBuffer {
    /// Allocate a zero-filled buffer of `len` positions for `dtype`.
    pub fn zeros(dtype: Dtype, len: usize) -> Result<Self> {
        Ok(match dtype {
            Dtype::U8 => ColumnBuffer::U8(Array1::from_elem(len, 0u8)),
            Dtype::U16 => ColumnBuffer::U16(Array1::from_elem(len, 0u16)),
            Dtype::U32 => ColumnBuffer::U32(Array1::from_elem(len, 0u32)),
            Dtype::I8 => ColumnBuffer::I8(Array1::from_elem(len, 0i8)),
            Dtype::I16 => ColumnBuffer::I16(Array1::from_elem(len, 0i16)),
            Dtype::I32 => ColumnBuffer::I32(Array1::from_elem(len, 0i32)),
            Dtype::F32 => ColumnBuffer::F32(Array1::from_elem(len, 0f32)),
            Dtype::F64 => ColumnBuffer::F64(Array1::from_elem(len, 0f64)),
            Dtype::Bool => ColumnBuffer::Bool(Array1::from_elem(len, false)),
        })
    }

    /// A mutable sink over the half-open sub-range `[lo, hi)` of this buffer.
    pub fn sink_slice(&mut self, lo: usize, hi: usize) -> ColumnSinkMut<'_> {
        match self {
            ColumnBuffer::U8(a) => ColumnSinkMut::U8(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::U16(a) => ColumnSinkMut::U16(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::U32(a) => ColumnSinkMut::U32(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::I8(a) => ColumnSinkMut::I8(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::I16(a) => ColumnSinkMut::I16(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::I32(a) => ColumnSinkMut::I32(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::F32(a) => ColumnSinkMut::F32(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::F64(a) => ColumnSinkMut::F64(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::Bool(a) => ColumnSinkMut::Bool(a.slice_mut(ndarray::s![lo..hi])),
        }
    }

    pub(crate) fn write_to_track(self, track: &Track, start: u64, end: u64) -> crate::Result<()> {
        match self {
            ColumnBuffer::U8(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::U16(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::U32(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::I8(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::I16(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::I32(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::F32(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::F64(values) => track.write_flat(start, end, values.into_dyn()),
            ColumnBuffer::Bool(values) => track.write_flat(start, end, values.into_dyn()),
        }
    }
}

/// One rank-2 scratch tile for one field over several source columns.
pub enum MatrixBuffer {
    U8(Array2<u8>),
    U16(Array2<u16>),
    U32(Array2<u32>),
    I8(Array2<i8>),
    I16(Array2<i16>),
    I32(Array2<i32>),
    F32(Array2<f32>),
    F64(Array2<f64>),
    Bool(Array2<bool>),
}

impl MatrixBuffer {
    pub fn zeros(dtype: Dtype, rows: usize, columns: usize) -> Result<Self> {
        Ok(match dtype {
            Dtype::U8 => MatrixBuffer::U8(Array2::from_elem((rows, columns), 0u8)),
            Dtype::U16 => MatrixBuffer::U16(Array2::from_elem((rows, columns), 0u16)),
            Dtype::U32 => MatrixBuffer::U32(Array2::from_elem((rows, columns), 0u32)),
            Dtype::I8 => MatrixBuffer::I8(Array2::from_elem((rows, columns), 0i8)),
            Dtype::I16 => MatrixBuffer::I16(Array2::from_elem((rows, columns), 0i16)),
            Dtype::I32 => MatrixBuffer::I32(Array2::from_elem((rows, columns), 0i32)),
            Dtype::F32 => MatrixBuffer::F32(Array2::from_elem((rows, columns), 0f32)),
            Dtype::F64 => MatrixBuffer::F64(Array2::from_elem((rows, columns), 0f64)),
            Dtype::Bool => MatrixBuffer::Bool(Array2::from_elem((rows, columns), false)),
        })
    }

    pub fn sink_column_slice(&mut self, rows: Range<usize>, column: usize) -> ColumnSinkMut<'_> {
        match self {
            MatrixBuffer::U8(a) => ColumnSinkMut::U8(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::U16(a) => ColumnSinkMut::U16(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::U32(a) => ColumnSinkMut::U32(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::I8(a) => ColumnSinkMut::I8(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::I16(a) => ColumnSinkMut::I16(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::I32(a) => ColumnSinkMut::I32(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::F32(a) => ColumnSinkMut::F32(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::F64(a) => ColumnSinkMut::F64(a.slice_mut(ndarray::s![rows, column])),
            MatrixBuffer::Bool(a) => ColumnSinkMut::Bool(a.slice_mut(ndarray::s![rows, column])),
        }
    }

    /// One disjoint mutable sink per column over the given row range, so a
    /// single reader can fill every column of one tile in one pass.
    pub fn sink_columns(&mut self, rows: Range<usize>) -> Vec<ColumnSinkMut<'_>> {
        match self {
            MatrixBuffer::U8(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::U8)
                .collect(),
            MatrixBuffer::U16(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::U16)
                .collect(),
            MatrixBuffer::U32(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::U32)
                .collect(),
            MatrixBuffer::I8(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::I8)
                .collect(),
            MatrixBuffer::I16(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::I16)
                .collect(),
            MatrixBuffer::I32(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::I32)
                .collect(),
            MatrixBuffer::F32(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::F32)
                .collect(),
            MatrixBuffer::F64(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::F64)
                .collect(),
            MatrixBuffer::Bool(a) => collect_column_sinks(a.slice_mut(ndarray::s![rows, ..]))
                .into_iter()
                .map(ColumnSinkMut::Bool)
                .collect(),
        }
    }

    pub(crate) fn write_to_track(
        self,
        track: &Track,
        position: Range<u64>,
        columns: Range<u64>,
    ) -> crate::Result<()> {
        match self {
            MatrixBuffer::U8(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::U16(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::U32(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::I8(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::I16(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::I32(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::F32(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::F64(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
            MatrixBuffer::Bool(values) => {
                track.write_flat_columns(position, columns, values.into_dyn())
            }
        }
    }
}

/// Split a rank-2 view into one rank-1 view per column, all borrowing the
/// original lifetime (repeated `split_at` sidesteps the borrow checker's
/// one-`&mut`-at-a-time rule for a plain iterator).
fn collect_column_sinks<T>(mut view: ArrayViewMut2<'_, T>) -> Vec<ArrayViewMut1<'_, T>> {
    let mut sinks = Vec::with_capacity(view.ncols());
    while view.ncols() > 0 {
        let (column, rest) = view.split_at(Axis(1), 1);
        sinks.push(column.remove_axis(Axis(1)));
        view = rest;
    }
    sinks
}

/// A borrowing, dtype-tagged view a reader fills. The reader hands raw cells to
/// `fill_run`; the parse into the concrete dtype happens here, so the reader
/// stays dtype-blind.
pub enum ColumnSinkMut<'a> {
    U8(ArrayViewMut1<'a, u8>),
    U16(ArrayViewMut1<'a, u16>),
    U32(ArrayViewMut1<'a, u32>),
    I8(ArrayViewMut1<'a, i8>),
    I16(ArrayViewMut1<'a, i16>),
    I32(ArrayViewMut1<'a, i32>),
    F32(ArrayViewMut1<'a, f32>),
    F64(ArrayViewMut1<'a, f64>),
    Bool(ArrayViewMut1<'a, bool>),
}

impl ColumnSinkMut<'_> {
    /// Bulk copy for readers that accumulate in their own contiguous scratch.
    pub fn assign_i32(&mut self, src: &[i32]) -> Result<()> {
        match self {
            ColumnSinkMut::I32(v) => {
                if v.len() != src.len() {
                    return Err(ReaderError::Other(anyhow::anyhow!(
                        "assign_i32 length {} != sink {}",
                        src.len(),
                        v.len()
                    )));
                }
                v.assign(&ArrayView1::from(src));
                Ok(())
            }
            _ => Err(ReaderError::Other(anyhow::anyhow!(
                "assign_i32 on non-i32 sink"
            ))),
        }
    }

    /// Parse `cell` into this sink's dtype and write it across `[lo, hi)`.
    pub fn fill_run(&mut self, lo: usize, hi: usize, cell: &str) -> Result<()> {
        match self {
            ColumnSinkMut::U8(v) => v
                .slice_mut(ndarray::s![lo..hi])
                .fill(parse_cell(cell, "uint8")?),
            ColumnSinkMut::U16(v) => v
                .slice_mut(ndarray::s![lo..hi])
                .fill(parse_cell(cell, "uint16")?),
            ColumnSinkMut::U32(v) => v
                .slice_mut(ndarray::s![lo..hi])
                .fill(parse_cell(cell, "uint32")?),
            ColumnSinkMut::I8(v) => v
                .slice_mut(ndarray::s![lo..hi])
                .fill(parse_cell(cell, "int8")?),
            ColumnSinkMut::I16(v) => v
                .slice_mut(ndarray::s![lo..hi])
                .fill(parse_cell(cell, "int16")?),
            ColumnSinkMut::I32(v) => {
                v.slice_mut(ndarray::s![lo..hi])
                    .fill(parse_cell(cell, "int32")?);
            }
            ColumnSinkMut::F32(v) => {
                v.slice_mut(ndarray::s![lo..hi])
                    .fill(parse_cell(cell, "float32")?);
            }
            ColumnSinkMut::F64(v) => {
                v.slice_mut(ndarray::s![lo..hi])
                    .fill(parse_cell(cell, "float64")?);
            }
            ColumnSinkMut::Bool(v) => {
                let x = parse_bool(cell).ok_or_else(|| {
                    ReaderError::Other(anyhow::anyhow!(
                        "parse {cell:?} as bool (want 0/1/true/false)"
                    ))
                })?;
                v.slice_mut(ndarray::s![lo..hi]).fill(x);
            }
        }
        Ok(())
    }
}

fn parse_cell<T: std::str::FromStr>(cell: &str, dtype: &str) -> Result<T>
where
    T::Err: Display,
{
    cell.trim()
        .parse()
        .map_err(|error| ReaderError::Other(anyhow::anyhow!("parse {cell:?} as {dtype}: {error}")))
}

/// BED boolean columns are usually `0`/`1`; also accept `true`/`false`.
fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "1" | "true" | "True" | "TRUE" => Some(true),
        "0" | "false" | "False" | "FALSE" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_run_writes_parsed_values_over_subrange() {
        let mut buf = ColumnBuffer::zeros(Dtype::I32, 10).unwrap();
        {
            let mut s = buf.sink_slice(2, 6);
            s.fill_run(0, 4, "42").unwrap();
        }
        match buf {
            ColumnBuffer::I32(a) => {
                assert_eq!(a[1], 0);
                assert!(a.slice(ndarray::s![2..6]).iter().all(|&v| v == 42));
                assert_eq!(a[6], 0);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn bool_accepts_zero_one_and_words() {
        let mut buf = ColumnBuffer::zeros(Dtype::Bool, 4).unwrap();
        {
            let mut s = buf.sink_slice(0, 4);
            s.fill_run(0, 2, "1").unwrap();
            s.fill_run(2, 4, "false").unwrap();
        }
        match buf {
            ColumnBuffer::Bool(a) => assert_eq!(a.to_vec(), vec![true, true, false, false]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn f32_parse_error_is_reported() {
        let mut buf = ColumnBuffer::zeros(Dtype::F32, 3).unwrap();
        let mut s = buf.sink_slice(0, 3);
        assert!(s.fill_run(0, 3, "notanumber").is_err());
    }

    #[test]
    fn zeros_supports_every_pbz_dtype() {
        for dtype in [
            Dtype::U8,
            Dtype::U16,
            Dtype::U32,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::F32,
            Dtype::F64,
            Dtype::Bool,
        ] {
            assert!(ColumnBuffer::zeros(dtype, 4).is_ok());
        }
    }

    #[test]
    fn sinks_parse_all_pbz_dtypes() {
        let cases = [
            (Dtype::U8, "255"),
            (Dtype::U16, "65535"),
            (Dtype::U32, "4294967295"),
            (Dtype::I8, "-128"),
            (Dtype::I16, "-32768"),
            (Dtype::I32, "-2147483648"),
            (Dtype::F32, "1.5"),
            (Dtype::F64, "1.5"),
            (Dtype::Bool, "true"),
        ];
        for (dtype, cell) in cases {
            let mut buffer = ColumnBuffer::zeros(dtype, 1).unwrap();
            buffer.sink_slice(0, 1).fill_run(0, 1, cell).unwrap();
            match (dtype, buffer) {
                (Dtype::U8, ColumnBuffer::U8(values)) => assert_eq!(values[0], 255),
                (Dtype::U16, ColumnBuffer::U16(values)) => assert_eq!(values[0], 65535),
                (Dtype::U32, ColumnBuffer::U32(values)) => {
                    assert_eq!(values[0], 4_294_967_295)
                }
                (Dtype::I8, ColumnBuffer::I8(values)) => assert_eq!(values[0], -128),
                (Dtype::I16, ColumnBuffer::I16(values)) => assert_eq!(values[0], -32768),
                (Dtype::I32, ColumnBuffer::I32(values)) => {
                    assert_eq!(values[0], -2_147_483_648)
                }
                (Dtype::F32, ColumnBuffer::F32(values)) => assert_eq!(values[0], 1.5),
                (Dtype::F64, ColumnBuffer::F64(values)) => assert_eq!(values[0], 1.5),
                (Dtype::Bool, ColumnBuffer::Bool(values)) => assert!(values[0]),
                (actual, _) => panic!("wrong buffer variant for {actual}"),
            }
        }
    }

    #[test]
    fn matrix_buffer_fills_one_source_column_only() {
        let mut buffer = MatrixBuffer::zeros(Dtype::U16, 3, 2).unwrap();
        buffer
            .sink_column_slice(0..3, 1)
            .fill_run(0, 3, "255")
            .unwrap();
        match buffer {
            MatrixBuffer::U16(values) => {
                assert_eq!(values[[0, 0]], 0);
                assert_eq!(values[[2, 1]], 255);
            }
            _ => panic!("wrong matrix variant"),
        }
    }
}
