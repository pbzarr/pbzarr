//! `pbz stat` compute engine: region statistics over one track.

use std::str::FromStr;

use crate::error::{PbzError, Result};

/// The statistic one `stat` run computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatKind {
    Mean,
    Min,
    Max,
    Median,
    Hist,
}

impl StatKind {
    pub fn name(self) -> &'static str {
        match self {
            StatKind::Mean => "mean",
            StatKind::Min => "min",
            StatKind::Max => "max",
            StatKind::Median => "median",
            StatKind::Hist => "hist",
        }
    }
}

impl FromStr for StatKind {
    type Err = PbzError;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "mean" => Ok(StatKind::Mean),
            "min" => Ok(StatKind::Min),
            "max" => Ok(StatKind::Max),
            "median" => Ok(StatKind::Median),
            "hist" => Ok(StatKind::Hist),
            other => Err(PbzError::Metadata(format!(
                "unknown stat {other:?} (expected mean, min, max, median, hist)"
            ))),
        }
    }
}

/// Mergeable per-sample state: identity from the constructor, `merge` takes
/// the other accumulator by value so partials reduce without locks.
pub(crate) trait Accumulator: Send {
    type Value: Copy;
    type Output;
    fn update(&mut self, value: Self::Value);
    fn merge(&mut self, other: Self);
    fn finalize(self) -> Self::Output;
}

#[derive(Debug, Default)]
pub(crate) struct MeanAcc {
    sum: f64,
    count: u64,
}

impl Accumulator for MeanAcc {
    type Value = f64;
    type Output = f64;

    fn update(&mut self, value: f64) {
        if !value.is_nan() {
            self.sum += value;
            self.count += 1;
        }
    }

    fn merge(&mut self, other: Self) {
        self.sum += other.sum;
        self.count += other.count;
    }

    fn finalize(self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }
}

#[derive(Debug)]
pub(crate) struct MinAcc(f64);

impl Default for MinAcc {
    fn default() -> Self {
        Self(f64::NAN)
    }
}

impl Accumulator for MinAcc {
    type Value = f64;
    type Output = f64;

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn update(&mut self, value: f64) {
        if !value.is_nan() && !(self.0 <= value) {
            self.0 = value;
        }
    }

    fn merge(&mut self, other: Self) {
        self.update(other.0);
    }

    fn finalize(self) -> f64 {
        self.0
    }
}

#[derive(Debug)]
pub(crate) struct MaxAcc(f64);

impl Default for MaxAcc {
    fn default() -> Self {
        Self(f64::NAN)
    }
}

impl Accumulator for MaxAcc {
    type Value = f64;
    type Output = f64;

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn update(&mut self, value: f64) {
        if !value.is_nan() && !(self.0 >= value) {
            self.0 = value;
        }
    }

    fn merge(&mut self, other: Self) {
        self.update(other.0);
    }

    fn finalize(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_kind_parses_names() {
        assert_eq!("mean".parse::<StatKind>().unwrap(), StatKind::Mean);
        assert_eq!("hist".parse::<StatKind>().unwrap(), StatKind::Hist);
        assert!("average".parse::<StatKind>().is_err());
    }

    #[test]
    fn mean_acc_merges_and_skips_nan() {
        let mut a = MeanAcc::default();
        for v in [1.0, 2.0, f64::NAN] {
            a.update(v);
        }
        let mut b = MeanAcc::default();
        b.update(3.0);
        a.merge(b);
        assert_eq!(a.finalize(), 2.0);
        assert!(MeanAcc::default().finalize().is_nan());
    }

    #[test]
    fn min_max_accs_skip_nan_and_merge() {
        let mut lo = MinAcc::default();
        let mut hi = MaxAcc::default();
        for v in [f64::NAN, 3.0, -1.0, 2.0] {
            lo.update(v);
            hi.update(v);
        }
        let mut lo2 = MinAcc::default();
        lo2.update(-5.0);
        lo.merge(lo2);
        assert_eq!(lo.finalize(), -5.0);
        assert_eq!(hi.finalize(), 3.0);
        assert!(MinAcc::default().finalize().is_nan());
    }
}
