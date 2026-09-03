//! `ColumnSinkMut`: a borrowing, dtype-tagged view whose [`fill_run`] parses
//! raw text cells into the sink's dtype, so text readers (BED) stay
//! dtype-blind.
//!
//! [`fill_run`]: ColumnSinkMut::fill_run

use std::fmt::Display;

use ndarray::ArrayViewMut1;

use crate::io::error::{ReaderError, Result};

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
    use ndarray::Array1;

    #[test]
    fn fill_run_writes_parsed_values_over_subrange() {
        let mut buf = Array1::<i32>::zeros(10);
        ColumnSinkMut::I32(buf.slice_mut(ndarray::s![2..6]))
            .fill_run(0, 4, "42")
            .unwrap();
        assert_eq!(buf[1], 0);
        assert!(buf.slice(ndarray::s![2..6]).iter().all(|&v| v == 42));
        assert_eq!(buf[6], 0);
    }

    #[test]
    fn f32_parse_error_is_reported() {
        let mut buf = Array1::<f32>::zeros(3);
        let mut s = ColumnSinkMut::F32(buf.view_mut());
        assert!(s.fill_run(0, 3, "notanumber").is_err());
    }

    #[test]
    fn sinks_parse_all_pbz_dtypes() {
        macro_rules! check_parse {
            ($variant:ident, $ty:ty, $cell:expr, $expected:expr) => {{
                let mut buf = Array1::<$ty>::from_elem(1, <$ty>::default());
                ColumnSinkMut::$variant(buf.view_mut())
                    .fill_run(0, 1, $cell)
                    .unwrap();
                assert_eq!(buf[0], $expected);
            }};
        }
        check_parse!(U8, u8, "255", 255);
        check_parse!(U16, u16, "65535", 65535);
        check_parse!(U32, u32, "4294967295", 4_294_967_295);
        check_parse!(I8, i8, "-128", -128);
        check_parse!(I16, i16, "-32768", -32768);
        check_parse!(I32, i32, "-2147483648", -2_147_483_648);
        check_parse!(F32, f32, "1.5", 1.5);
        check_parse!(F64, f64, "1.5", 1.5);
        check_parse!(Bool, bool, "true", true);
        check_parse!(Bool, bool, "1", true);
        check_parse!(Bool, bool, "0", false);
    }
}
