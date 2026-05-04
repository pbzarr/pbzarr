//! Pure transform: stream of `IntervalRecord` -> dense `Array1<T>` over a
//! region, with per-dtype value casts.

use crate::io::bed_error::BedError;
use crate::io::bed_reader::{IntervalRecord, IntervalValue};
use ndarray::Array1;
use std::path::Path;

/// Expand a sorted, non-overlapping interval iterator into a dense per-base
/// `Array1<T>` for the half-open region `[region_start, region_end)`.
///
/// `fill` is the value for positions not covered by any interval. `cast`
/// converts each interval's `IntervalValue` into the target type, and may
/// reject (e.g. on out-of-range numeric).
///
/// Defensive overlap detection: if an interval's start lies before the
/// previous interval's end (after clipping to the region), this returns
/// `BedError::Overlap`. Primary detection happens in the load phase; this
/// catches caller bugs.
pub(crate) fn expand_intervals_into<'a, T>(
    intervals: impl Iterator<Item = &'a IntervalRecord>,
    contig: &str,
    region_start: u64,
    region_end: u64,
    fill: T,
    cast: impl Fn(&IntervalValue) -> Result<T, BedError>,
    source_path: &Path,
) -> Result<Array1<T>, BedError>
where
    T: Copy + Clone,
{
    assert!(region_end >= region_start, "region_end < region_start");
    let len = (region_end - region_start) as usize;
    let mut arr = Array1::<T>::from_elem(len, fill);

    let mut prev_end_in_region: u64 = region_start;
    for r in intervals {
        if r.end <= region_start || r.start >= region_end {
            continue;
        }
        let s = r.start.max(region_start);
        let e = r.end.min(region_end);
        if s < prev_end_in_region {
            return Err(BedError::Overlap {
                path: source_path.to_path_buf(),
                line: 0,
                contig: contig.to_string(),
                this_start: r.start,
                this_end: r.end,
                prev_end: prev_end_in_region,
            });
        }
        let v = cast(&r.value)?;
        let s_idx = (s - region_start) as usize;
        let e_idx = (e - region_start) as usize;
        arr.slice_mut(ndarray::s![s_idx..e_idx]).fill(v);
        prev_end_in_region = e;
    }
    Ok(arr)
}

pub(crate) fn cast_mask_to_bool(_v: &IntervalValue) -> Result<bool, BedError> {
    Ok(true)
}

fn numeric(v: &IntervalValue) -> f64 {
    match v {
        IntervalValue::Numeric(x) => *x,
        IntervalValue::Mask => unreachable!("numeric cast called on Mask flavor"),
    }
}

fn out_of_range(v: f64, dtype: &'static str) -> BedError {
    BedError::ValueOutOfRange {
        path: std::path::PathBuf::new(),
        line: 0,
        value: v,
        dtype,
    }
}

pub(crate) fn cast_f64_to_f32(v: &IntervalValue) -> Result<f32, BedError> {
    let x = numeric(v);
    if !x.is_finite() {
        // Allow NaN/inf to pass through — bedGraph allows them.
        return Ok(x as f32);
    }
    if x > f32::MAX as f64 || x < f32::MIN as f64 {
        return Err(out_of_range(x, "float32"));
    }
    Ok(x as f32)
}

pub(crate) fn cast_f64_to_f64(v: &IntervalValue) -> Result<f64, BedError> {
    Ok(numeric(v))
}

fn cast_f64_to_uint<U>(v: &IntervalValue, max: f64, dtype: &'static str) -> Result<U, BedError>
where
    U: TryFrom<u64>,
    <U as TryFrom<u64>>::Error: std::fmt::Debug,
{
    let x = numeric(v);
    if !x.is_finite() || x < 0.0 || x > max || x.fract() != 0.0 {
        return Err(out_of_range(x, dtype));
    }
    Ok(U::try_from(x as u64).expect("range checked above"))
}

fn cast_f64_to_int<I>(
    v: &IntervalValue,
    min: f64,
    max: f64,
    dtype: &'static str,
) -> Result<I, BedError>
where
    I: TryFrom<i64>,
    <I as TryFrom<i64>>::Error: std::fmt::Debug,
{
    let x = numeric(v);
    if !x.is_finite() || x < min || x > max || x.fract() != 0.0 {
        return Err(out_of_range(x, dtype));
    }
    Ok(I::try_from(x as i64).expect("range checked above"))
}

pub(crate) fn cast_f64_to_uint8(v: &IntervalValue) -> Result<u8, BedError> {
    cast_f64_to_uint::<u8>(v, u8::MAX as f64, "uint8")
}
pub(crate) fn cast_f64_to_uint16(v: &IntervalValue) -> Result<u16, BedError> {
    cast_f64_to_uint::<u16>(v, u16::MAX as f64, "uint16")
}
pub(crate) fn cast_f64_to_uint32(v: &IntervalValue) -> Result<u32, BedError> {
    cast_f64_to_uint::<u32>(v, u32::MAX as f64, "uint32")
}
pub(crate) fn cast_f64_to_int8(v: &IntervalValue) -> Result<i8, BedError> {
    cast_f64_to_int::<i8>(v, i8::MIN as f64, i8::MAX as f64, "int8")
}
pub(crate) fn cast_f64_to_int16(v: &IntervalValue) -> Result<i16, BedError> {
    cast_f64_to_int::<i16>(v, i16::MIN as f64, i16::MAX as f64, "int16")
}
pub(crate) fn cast_f64_to_int32(v: &IntervalValue) -> Result<i32, BedError> {
    cast_f64_to_int::<i32>(v, i32::MIN as f64, i32::MAX as f64, "int32")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn rec(start: u64, end: u64) -> IntervalRecord {
        IntervalRecord {
            start,
            end,
            value: IntervalValue::Mask,
        }
    }

    fn p() -> PathBuf {
        PathBuf::from("/dev/null")
    }

    #[test]
    fn empty_input_yields_fill_only() {
        let arr = expand_intervals_into::<bool>(
            std::iter::empty(),
            "chr1",
            10,
            20,
            false,
            cast_mask_to_bool,
            &p(),
        )
        .unwrap();
        assert_eq!(arr.len(), 10);
        assert!(arr.iter().all(|&v| !v));
    }

    #[test]
    fn single_interval_filled() {
        let recs = vec![rec(12, 15)];
        let arr = expand_intervals_into::<bool>(
            recs.iter(),
            "chr1",
            10,
            20,
            false,
            cast_mask_to_bool,
            &p(),
        )
        .unwrap();
        let expected = [
            false, false, true, true, true, false, false, false, false, false,
        ];
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, expected[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn interval_clipped_to_region() {
        let recs = vec![rec(5, 12), rec(18, 25)];
        let arr = expand_intervals_into::<bool>(
            recs.iter(),
            "chr1",
            10,
            20,
            false,
            cast_mask_to_bool,
            &p(),
        )
        .unwrap();
        let expected = [
            true, true, false, false, false, false, false, false, true, true,
        ];
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, expected[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn intervals_outside_region_skipped() {
        let recs = vec![rec(0, 5), rec(25, 30)];
        let arr = expand_intervals_into::<bool>(
            recs.iter(),
            "chr1",
            10,
            20,
            false,
            cast_mask_to_bool,
            &p(),
        )
        .unwrap();
        assert!(arr.iter().all(|&v| !v));
    }

    #[test]
    fn overlap_within_region_is_error() {
        let recs = vec![rec(10, 15), rec(14, 18)];
        let err = expand_intervals_into::<bool>(
            recs.iter(),
            "chr1",
            10,
            20,
            false,
            cast_mask_to_bool,
            &p(),
        )
        .unwrap_err();
        assert!(matches!(err, BedError::Overlap { .. }));
    }

    fn nrec(start: u64, end: u64, v: f64) -> IntervalRecord {
        IntervalRecord {
            start,
            end,
            value: IntervalValue::Numeric(v),
        }
    }

    #[test]
    fn cast_f64_to_f32_lossless_in_range() {
        let r = IntervalValue::Numeric(1.5);
        assert_eq!(super::cast_f64_to_f32(&r).unwrap(), 1.5_f32);
    }

    #[test]
    fn cast_f64_to_f32_overflow_rejected() {
        let too_big = IntervalValue::Numeric(1.0e40);
        assert!(matches!(
            super::cast_f64_to_f32(&too_big),
            Err(BedError::ValueOutOfRange {
                dtype: "float32",
                ..
            })
        ));
    }

    #[test]
    fn cast_f64_to_uint8_in_range() {
        let r = IntervalValue::Numeric(200.0);
        assert_eq!(super::cast_f64_to_uint8(&r).unwrap(), 200u8);
    }

    #[test]
    fn cast_f64_to_uint8_negative_rejected() {
        let r = IntervalValue::Numeric(-1.0);
        assert!(matches!(
            super::cast_f64_to_uint8(&r),
            Err(BedError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn cast_f64_to_uint8_overflow_rejected() {
        let r = IntervalValue::Numeric(300.0);
        assert!(matches!(
            super::cast_f64_to_uint8(&r),
            Err(BedError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn cast_f64_to_uint8_fractional_rejected() {
        let r = IntervalValue::Numeric(2.5);
        assert!(matches!(
            super::cast_f64_to_uint8(&r),
            Err(BedError::ValueOutOfRange { .. })
        ));
    }

    #[test]
    fn cast_f64_to_int16_negative_ok() {
        let r = IntervalValue::Numeric(-100.0);
        assert_eq!(super::cast_f64_to_int16(&r).unwrap(), -100i16);
    }

    #[test]
    fn cast_mask_into_numeric_dtype_panics_or_errors() {
        let r = IntervalValue::Numeric(0.0);
        assert!(super::cast_f64_to_uint32(&r).is_ok());
    }

    #[test]
    fn expand_with_numeric_fills_correctly() {
        let recs = vec![nrec(10, 13, 5.0), nrec(15, 18, 7.0)];
        let arr = expand_intervals_into::<u8>(
            recs.iter(),
            "chr1",
            10,
            20,
            0u8,
            super::cast_f64_to_uint8,
            &p(),
        )
        .unwrap();
        let expected = [5, 5, 5, 0, 0, 7, 7, 7, 0, 0];
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, expected[i], "mismatch at index {i}");
        }
    }
}
