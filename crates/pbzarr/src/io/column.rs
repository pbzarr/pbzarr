//! Dtype-erased column buffers for the multi-column import pipeline. One BED
//! file feeds several tracks of different dtypes in a single decode pass, so
//! the pipeline holds one buffer per track behind a closed enum rather than a
//! single generic `T`. Only the three dtypes a multi-column import can target
//! (`i32` / `f32` / `bool`) are represented.

use ndarray::{Array1, ArrayViewMut1};

use crate::io::dtype::Dtype;
use crate::io::error::{ReaderError, Result};

/// One track's owning scratch buffer for a chunk region, tagged by dtype.
pub enum ColumnBuffer {
    I32(Array1<i32>),
    F32(Array1<f32>),
    Bool(Array1<bool>),
}

impl ColumnBuffer {
    /// Allocate a zero-filled buffer of `len` positions for `dtype`. Errors on
    /// any dtype outside the multi-import set.
    pub fn zeros(dtype: Dtype, len: usize) -> Result<Self> {
        Ok(match dtype {
            Dtype::I32 => ColumnBuffer::I32(Array1::from_elem(len, 0i32)),
            Dtype::F32 => ColumnBuffer::F32(Array1::from_elem(len, 0f32)),
            Dtype::Bool => ColumnBuffer::Bool(Array1::from_elem(len, false)),
            other => {
                return Err(ReaderError::Other(anyhow::anyhow!(
                    "multi-column import cannot target dtype {other}"
                )));
            }
        })
    }

    /// A mutable sink over the half-open sub-range `[lo, hi)` of this buffer.
    pub fn sink_slice(&mut self, lo: usize, hi: usize) -> ColumnSinkMut<'_> {
        match self {
            ColumnBuffer::I32(a) => ColumnSinkMut::I32(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::F32(a) => ColumnSinkMut::F32(a.slice_mut(ndarray::s![lo..hi])),
            ColumnBuffer::Bool(a) => ColumnSinkMut::Bool(a.slice_mut(ndarray::s![lo..hi])),
        }
    }
}

/// A borrowing, dtype-tagged view a reader fills. The reader hands raw cells to
/// `fill_run`; the parse into the concrete dtype happens here, so the reader
/// stays dtype-blind.
pub enum ColumnSinkMut<'a> {
    I32(ArrayViewMut1<'a, i32>),
    F32(ArrayViewMut1<'a, f32>),
    Bool(ArrayViewMut1<'a, bool>),
}

impl ColumnSinkMut<'_> {
    /// Parse `cell` into this sink's dtype and write it across `[lo, hi)`.
    pub fn fill_run(&mut self, lo: usize, hi: usize, cell: &str) -> Result<()> {
        match self {
            ColumnSinkMut::I32(v) => {
                let x: i32 = cell.trim().parse().map_err(|e| {
                    ReaderError::Other(anyhow::anyhow!("parse {cell:?} as int32: {e}"))
                })?;
                v.slice_mut(ndarray::s![lo..hi]).fill(x);
            }
            ColumnSinkMut::F32(v) => {
                let x: f32 = cell.trim().parse().map_err(|e| {
                    ReaderError::Other(anyhow::anyhow!("parse {cell:?} as float32: {e}"))
                })?;
                v.slice_mut(ndarray::s![lo..hi]).fill(x);
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
    fn zeros_rejects_unsupported_dtype() {
        assert!(ColumnBuffer::zeros(Dtype::U8, 4).is_err());
    }
}
