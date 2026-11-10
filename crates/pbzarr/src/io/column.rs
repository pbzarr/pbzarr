//! Text cell parsing for [`OutputSinkMut::fill_run`], so text readers (BED)
//! stay dtype-blind.

use std::fmt::Display;

use crate::io::Dtype;
use crate::io::error::{ReaderError, Result};
use crate::io::reader::OutputSinkMut;

impl OutputSinkMut<'_> {
    /// Parse `cell` into this sink's dtype and write it across `[lo, hi)`.
    pub fn fill_run(&mut self, lo: usize, hi: usize, cell: &str) -> Result<()> {
        macro_rules! fill_parsed {
            ($v:expr, $variant:ident) => {
                $v.slice_mut(ndarray::s![lo..hi])
                    .fill(parse_cell(cell, Dtype::$variant)?)
            };
        }
        match self {
            OutputSinkMut::U8(v) => fill_parsed!(v, U8),
            OutputSinkMut::U16(v) => fill_parsed!(v, U16),
            OutputSinkMut::U32(v) => fill_parsed!(v, U32),
            OutputSinkMut::U64(v) => fill_parsed!(v, U64),
            OutputSinkMut::I8(v) => fill_parsed!(v, I8),
            OutputSinkMut::I16(v) => fill_parsed!(v, I16),
            OutputSinkMut::I32(v) => fill_parsed!(v, I32),
            OutputSinkMut::I64(v) => fill_parsed!(v, I64),
            OutputSinkMut::F32(v) => fill_parsed!(v, F32),
            OutputSinkMut::F64(v) => fill_parsed!(v, F64),
            OutputSinkMut::Bool(v) => {
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

fn parse_cell<T: std::str::FromStr>(cell: &str, dtype: Dtype) -> Result<T>
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
        OutputSinkMut::I32(buf.slice_mut(ndarray::s![2..6]))
            .fill_run(0, 4, "42")
            .unwrap();
        assert_eq!(buf[1], 0);
        assert!(buf.slice(ndarray::s![2..6]).iter().all(|&v| v == 42));
        assert_eq!(buf[6], 0);
    }

    #[test]
    fn f32_parse_error_is_reported() {
        let mut buf = Array1::<f32>::zeros(3);
        let mut s = OutputSinkMut::F32(buf.view_mut());
        assert!(s.fill_run(0, 3, "notanumber").is_err());
    }

    #[test]
    fn sinks_parse_all_pbz_dtypes() {
        macro_rules! check_parse {
            ($variant:ident, $ty:ty, $cell:expr, $expected:expr) => {{
                let mut buf = Array1::<$ty>::from_elem(1, <$ty>::default());
                OutputSinkMut::$variant(buf.view_mut())
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
        check_parse!(U64, u64, "18446744073709551615", u64::MAX);
        check_parse!(I64, i64, "-9223372036854775808", i64::MIN);
        check_parse!(F32, f32, "1.5", 1.5);
        check_parse!(F64, f64, "1.5", 1.5);
        check_parse!(Bool, bool, "true", true);
        check_parse!(Bool, bool, "1", true);
        check_parse!(Bool, bool, "0", false);
    }
}
