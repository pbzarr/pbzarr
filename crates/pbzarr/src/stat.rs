//! `pbz stat` compute engine: region statistics over one track.

// Consumed by the stat engine in a later commit; drop with it.
#![allow(dead_code)]

use std::collections::BTreeMap;
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

#[derive(Debug)]
enum Counts {
    Dense { base: i64, counts: Vec<u64> },
    Sparse(BTreeMap<i64, u64>),
}

/// Exact value -> count map. Dense over the full dtype range for narrow
/// dtypes, sparse for i32/u32 where coverage clusters at small values.
#[derive(Debug)]
pub(crate) struct CountAcc {
    counts: Counts,
    total: u64,
}

impl CountAcc {
    pub(crate) fn dense(base: i64, len: usize) -> Self {
        Self {
            counts: Counts::Dense {
                base,
                counts: vec![0; len],
            },
            total: 0,
        }
    }

    pub(crate) fn sparse() -> Self {
        Self {
            counts: Counts::Sparse(BTreeMap::new()),
            total: 0,
        }
    }

    pub(crate) fn total(&self) -> u64 {
        self.total
    }

    pub(crate) fn count_of(&self, value: i64) -> u64 {
        match &self.counts {
            Counts::Dense { base, counts } => usize::try_from(value - base)
                .ok()
                .and_then(|i| counts.get(i).copied())
                .unwrap_or(0),
            Counts::Sparse(map) => map.get(&value).copied().unwrap_or(0),
        }
    }

    pub(crate) fn observed(&self) -> Vec<(i64, u64)> {
        match &self.counts {
            Counts::Dense { base, counts } => counts
                .iter()
                .enumerate()
                .filter(|&(_, c)| *c > 0)
                .map(|(i, &c)| (base + i as i64, c))
                .collect(),
            Counts::Sparse(map) => map.iter().map(|(&v, &c)| (v, c)).collect(),
        }
    }

    /// Lower of the two middle values for an even total.
    pub(crate) fn median(&self) -> i64 {
        let target = (self.total.saturating_sub(1)) / 2;
        let mut cumulative = 0u64;
        for (value, count) in self.observed() {
            cumulative += count;
            if cumulative > target {
                return value;
            }
        }
        0
    }
}

impl Accumulator for CountAcc {
    type Value = i64;
    type Output = CountAcc;

    fn update(&mut self, value: i64) {
        self.total += 1;
        match &mut self.counts {
            Counts::Dense { base, counts } => {
                let index = usize::try_from(value - *base).expect("value below dense base");
                counts[index] += 1;
            }
            Counts::Sparse(map) => *map.entry(value).or_insert(0) += 1,
        }
    }

    fn merge(&mut self, other: Self) {
        self.total += other.total;
        match (&mut self.counts, other.counts) {
            (Counts::Dense { counts, .. }, Counts::Dense { counts: rhs, .. }) => {
                for (slot, add) in counts.iter_mut().zip(rhs) {
                    *slot += add;
                }
            }
            (Counts::Sparse(map), Counts::Sparse(rhs)) => {
                for (value, count) in rhs {
                    *map.entry(value).or_insert(0) += count;
                }
            }
            _ => unreachable!("one run never mixes dense and sparse counts"),
        }
    }

    fn finalize(self) -> CountAcc {
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlatRange {
    pub region_pos: usize,
    pub lo: u64,
    pub hi: u64,
}

#[derive(Debug)]
pub(crate) struct Batch {
    pub span: std::ops::Range<u64>,
    pub items: Vec<FlatRange>,
}

/// Cut sorted flat ranges into pieces at chunk boundaries and pack
/// consecutive touched chunks into batches. Each chunk lands in exactly one
/// batch, so overlapping regions never decode a chunk twice.
pub(crate) fn plan_batches(ranges: &[FlatRange], chunk: u64, max_chunks: u64) -> Vec<Batch> {
    let max_chunks = max_chunks.max(1);
    let mut pieces: Vec<(u64, FlatRange)> = Vec::new();
    for r in ranges {
        let mut lo = r.lo;
        while lo < r.hi {
            let chunk_index = lo / chunk;
            let hi = r.hi.min((chunk_index + 1) * chunk);
            pieces.push((
                chunk_index,
                FlatRange {
                    region_pos: r.region_pos,
                    lo,
                    hi,
                },
            ));
            lo = hi;
        }
    }
    pieces.sort_by_key(|(c, p)| (*c, p.region_pos, p.lo));

    struct Open {
        first_chunk: u64,
        last_chunk: u64,
        batch: Batch,
    }
    let mut batches = Vec::new();
    let mut open: Option<Open> = None;
    for (chunk_index, piece) in pieces {
        let start_new = match &open {
            None => true,
            Some(o) => chunk_index > o.last_chunk + 1 || chunk_index - o.first_chunk >= max_chunks,
        };
        if start_new {
            if let Some(o) = open.take() {
                batches.push(o.batch);
            }
            open = Some(Open {
                first_chunk: chunk_index,
                last_chunk: chunk_index,
                batch: Batch {
                    span: piece.lo..piece.hi,
                    items: Vec::new(),
                },
            });
        }
        let o = open.as_mut().expect("batch opened above");
        o.last_chunk = chunk_index;
        o.batch.span.start = o.batch.span.start.min(piece.lo);
        o.batch.span.end = o.batch.span.end.max(piece.hi);
        o.batch.items.push(piece);
    }
    if let Some(o) = open {
        batches.push(o.batch);
    }
    batches
}

/// Union mask for hist: sort, then merge overlapping or touching ranges.
pub(crate) fn coalesce(ranges: &mut Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    ranges.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &(lo, hi) in ranges.iter() {
        match out.last_mut() {
            Some((_, last_hi)) if lo <= *last_hi => *last_hi = (*last_hi).max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
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

    #[test]
    fn count_acc_dense_median_is_lower_middle() {
        let mut acc = CountAcc::dense(0, 256);
        for v in [5, 5, 5, 7, 7] {
            acc.update(v);
        }
        assert_eq!(acc.total(), 5);
        assert_eq!(acc.median(), 5);
        acc.update(9);
        // even total: 5,5,5,7,7,9 -> lower middle is 5
        assert_eq!(acc.median(), 5);
    }

    #[test]
    fn count_acc_sparse_merges_and_lists_observed() {
        let mut a = CountAcc::sparse();
        a.update(1_000_000);
        a.update(-3);
        let mut b = CountAcc::sparse();
        b.update(-3);
        a.merge(b);
        assert_eq!(a.observed(), vec![(-3, 2), (1_000_000, 1)]);
        assert_eq!(a.count_of(-3), 2);
        assert_eq!(a.count_of(0), 0);
        assert_eq!(a.median(), -3);
    }

    #[test]
    fn count_acc_dense_negative_base() {
        let mut acc = CountAcc::dense(-128, 256);
        for v in [-128, -128, 127] {
            acc.update(v);
        }
        assert_eq!(acc.observed(), vec![(-128, 2), (127, 1)]);
        assert_eq!(acc.median(), -128);
    }

    fn range(region_pos: usize, lo: u64, hi: u64) -> FlatRange {
        FlatRange { region_pos, lo, hi }
    }

    fn touched_chunks(batch: &Batch, chunk: u64) -> std::collections::BTreeSet<u64> {
        batch
            .items
            .iter()
            .flat_map(|i| (i.lo / chunk)..=((i.hi - 1) / chunk))
            .collect()
    }

    #[test]
    fn planner_shares_one_chunk_among_small_regions() {
        let ranges: Vec<FlatRange> = (0..100)
            .map(|i| range(i, i as u64 * 5, i as u64 * 5 + 3))
            .collect();
        let batches = plan_batches(&ranges, 1000, 16);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].items.len(), 100);
        assert_eq!(batches[0].span, 0..498);
    }

    #[test]
    fn planner_splits_oversized_region_at_chunk_boundaries() {
        let batches = plan_batches(&[range(0, 0, 100)], 10, 3);
        // 10 chunks, at most 3 per batch -> 4 batches, chunks disjoint
        assert_eq!(batches.len(), 4);
        let mut seen = std::collections::BTreeSet::new();
        for batch in &batches {
            for c in touched_chunks(batch, 10) {
                assert!(seen.insert(c), "chunk {c} appears in two batches");
            }
        }
        assert_eq!(seen.len(), 10);
        let total: u64 = batches
            .iter()
            .flat_map(|b| &b.items)
            .map(|i| i.hi - i.lo)
            .sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn planner_breaks_at_chunk_gaps() {
        let ranges = [range(0, 5, 10), range(1, 90_000, 90_010)];
        let batches = plan_batches(&ranges, 100, 1_000_000);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].span, 5..10);
        assert_eq!(batches[1].span, 90_000..90_010);
    }

    #[test]
    fn planner_keeps_overlapping_regions_in_one_batch() {
        let ranges = [range(0, 0, 50), range(1, 0, 50), range(2, 30, 80)];
        let batches = plan_batches(&ranges, 1000, 16);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].span, 0..80);
        assert_eq!(batches[0].items.len(), 3);
    }

    #[test]
    fn coalesce_merges_overlaps() {
        let mut ranges = vec![(30, 80), (0, 50), (0, 50), (100, 110)];
        assert_eq!(coalesce(&mut ranges), vec![(0, 80), (100, 110)]);
    }

    #[test]
    fn planner_treats_zero_max_chunks_as_one() {
        let ranges = [range(0, 0, 50), range(1, 0, 50), range(2, 30, 80)];
        let batches = plan_batches(&ranges, 1000, 0);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].items.len(), 3);
    }
}
