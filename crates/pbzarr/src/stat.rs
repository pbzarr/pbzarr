//! `pbz stat` compute engine: region statistics over one track.

use std::collections::BTreeMap;
use std::str::FromStr;

use rayon::prelude::*;

use crate::error::{PbzError, Result};
use crate::genome::Region;
use crate::io::{Dtype, Numeric};
use crate::track::Track;

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
        let target = (self.total().saturating_sub(1)) / 2;
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
pub(crate) fn coalesce(ranges: &mut [(u64, u64)]) -> Vec<(u64, u64)> {
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

#[derive(Debug, Default)]
pub struct StatOptions {
    /// Sample subset for rank-2 tracks. `None` means all samples.
    pub columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatValue {
    Float(f64),
    Int(i64),
}

#[derive(Debug)]
pub struct HistTable {
    /// Observed values, ascending.
    pub values: Vec<i64>,
    /// counts[value_index][output_column_index]
    pub counts: Vec<Vec<u64>>,
}

#[derive(Debug)]
pub enum StatResult {
    /// rows[input_region_index][output_column_index]
    PerRegion(Vec<Vec<StatValue>>),
    Hist(HistTable),
}

#[derive(Debug)]
pub struct StatOutput {
    /// Selected sample names in output order; empty for rank-1 tracks.
    pub samples: Vec<String>,
    pub result: StatResult,
}

const BATCH_TARGET_BYTES: u64 = 64 << 20;

pub fn run(
    track: &Track,
    regions: &[Region],
    kind: StatKind,
    options: &StatOptions,
) -> Result<StatOutput> {
    let (selected, samples) = resolve_columns(track, options)?;
    let offsets = track.genome().offsets();
    let mut ranges: Vec<FlatRange> = regions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let base = offsets[r.contig.as_usize()] as u64;
            FlatRange {
                region_pos: i,
                lo: base + r.start,
                hi: base + r.end,
            }
        })
        .collect();
    if kind == StatKind::Hist {
        let mut flat: Vec<(u64, u64)> = ranges.iter().map(|r| (r.lo, r.hi)).collect();
        ranges = coalesce(&mut flat)
            .into_iter()
            .map(|(lo, hi)| FlatRange {
                region_pos: 0,
                lo,
                hi,
            })
            .collect();
    }
    ranges.sort_by_key(|r| (r.lo, r.hi));
    let n_regions = if kind == StatKind::Hist {
        usize::from(!ranges.is_empty())
    } else {
        regions.len()
    };
    let chunk = track.chunk_size()? as u64;
    let per_position = track.columns_count()? as u64 * dtype_size(track.dtype()) as u64;
    let max_chunks = (BATCH_TARGET_BYTES / (chunk * per_position).max(1)).max(1);
    let batches = plan_batches(&ranges, chunk, max_chunks);
    let result = dispatch(track, &batches, n_regions, &selected, kind)?;
    Ok(StatOutput { samples, result })
}

/// Selected column indices (empty for rank-1) plus their labels.
fn resolve_columns(track: &Track, options: &StatOptions) -> Result<(Vec<usize>, Vec<String>)> {
    if track.rank() == 1 {
        if options.columns.is_some() {
            return Err(PbzError::Metadata(format!(
                "track {:?} has no column axis",
                track.name()
            )));
        }
        return Ok((Vec::new(), Vec::new()));
    }
    let labels = track.column_labels()?;
    let Some(requested) = &options.columns else {
        return Ok(((0..labels.len()).collect(), labels));
    };
    let mut indices = Vec::with_capacity(requested.len());
    for label in requested {
        let index = labels.iter().position(|c| c == label).ok_or_else(|| {
            PbzError::Metadata(format!(
                "column {label:?} is not a column of track {:?} (available: {})",
                track.name(),
                labels.join(", ")
            ))
        })?;
        indices.push(index);
    }
    Ok((indices, requested.clone()))
}

fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::Bool | Dtype::U8 | Dtype::I8 => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

fn dispatch(
    track: &Track,
    batches: &[Batch],
    n_regions: usize,
    selected: &[usize],
    kind: StatKind,
) -> Result<StatResult> {
    match kind {
        StatKind::Mean | StatKind::Min | StatKind::Max => {
            let rows = match track.dtype() {
                Dtype::U8 => {
                    float_rows::<u8>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::U16 => {
                    float_rows::<u16>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::U32 => {
                    float_rows::<u32>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::I8 => {
                    float_rows::<i8>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::I16 => {
                    float_rows::<i16>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::I32 => {
                    float_rows::<i32>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::F32 => {
                    float_rows::<f32>(track, batches, n_regions, selected, kind, |v| v.into())
                }
                Dtype::F64 => float_rows::<f64>(track, batches, n_regions, selected, kind, |v| v),
                Dtype::Bool => float_rows::<bool>(track, batches, n_regions, selected, kind, |v| {
                    if v { 1.0 } else { 0.0 }
                }),
            }?;
            Ok(StatResult::PerRegion(rows))
        }
        StatKind::Median | StatKind::Hist => {
            let accs = match track.dtype() {
                Dtype::U8 => count_accs::<u8>(
                    track,
                    batches,
                    n_regions,
                    selected,
                    || {
                        if kind == StatKind::Hist {
                            CountAcc::dense(0, 256)
                        } else {
                            CountAcc::sparse()
                        }
                    },
                    |v| v.into(),
                ),
                Dtype::I8 => count_accs::<i8>(
                    track,
                    batches,
                    n_regions,
                    selected,
                    || {
                        if kind == StatKind::Hist {
                            CountAcc::dense(-128, 256)
                        } else {
                            CountAcc::sparse()
                        }
                    },
                    |v| v.into(),
                ),
                Dtype::U16 => count_accs::<u16>(
                    track,
                    batches,
                    n_regions,
                    selected,
                    || {
                        if kind == StatKind::Hist {
                            CountAcc::dense(0, 65_536)
                        } else {
                            CountAcc::sparse()
                        }
                    },
                    |v| v.into(),
                ),
                Dtype::I16 => count_accs::<i16>(
                    track,
                    batches,
                    n_regions,
                    selected,
                    || {
                        if kind == StatKind::Hist {
                            CountAcc::dense(-32_768, 65_536)
                        } else {
                            CountAcc::sparse()
                        }
                    },
                    |v| v.into(),
                ),
                Dtype::U32 => {
                    count_accs::<u32>(track, batches, n_regions, selected, CountAcc::sparse, |v| {
                        v.into()
                    })
                }
                Dtype::I32 => {
                    count_accs::<i32>(track, batches, n_regions, selected, CountAcc::sparse, |v| {
                        v.into()
                    })
                }
                Dtype::Bool => count_accs::<bool>(
                    track,
                    batches,
                    n_regions,
                    selected,
                    || {
                        if kind == StatKind::Hist {
                            CountAcc::dense(0, 2)
                        } else {
                            CountAcc::sparse()
                        }
                    },
                    |v| v.into(),
                ),
                Dtype::F32 => Err(PbzError::InvalidDtype {
                    dtype: format!(
                        "{} needs an integer or bool track, but track {:?} is f32",
                        kind.name(),
                        track.name(),
                    ),
                }),
                Dtype::F64 => Err(PbzError::InvalidDtype {
                    dtype: format!(
                        "{} needs an integer or bool track, but track {:?} is f64",
                        kind.name(),
                        track.name(),
                    ),
                }),
            }?;
            if kind == StatKind::Median {
                let rows = accs
                    .into_iter()
                    .map(|per_sample| {
                        per_sample
                            .into_iter()
                            .map(|acc| StatValue::Int(acc.median()))
                            .collect()
                    })
                    .collect();
                Ok(StatResult::PerRegion(rows))
            } else {
                Ok(StatResult::Hist(hist_table(accs)))
            }
        }
    }
}

fn float_rows<T: Numeric>(
    track: &Track,
    batches: &[Batch],
    n_regions: usize,
    selected: &[usize],
    kind: StatKind,
    convert: fn(T) -> f64,
) -> Result<Vec<Vec<StatValue>>> {
    let integer = !matches!(track.dtype(), Dtype::F32 | Dtype::F64);
    let extreme = |v: f64| {
        if integer && v.is_finite() {
            StatValue::Int(v as i64)
        } else {
            StatValue::Float(v)
        }
    };
    let rows = match kind {
        StatKind::Mean => finalize_rows(
            accumulate::<T, MeanAcc, _, _>(
                track,
                batches,
                n_regions,
                selected,
                MeanAcc::default,
                convert,
            )?,
            StatValue::Float,
        ),
        StatKind::Min => finalize_rows(
            accumulate::<T, MinAcc, _, _>(
                track,
                batches,
                n_regions,
                selected,
                MinAcc::default,
                convert,
            )?,
            extreme,
        ),
        StatKind::Max => finalize_rows(
            accumulate::<T, MaxAcc, _, _>(
                track,
                batches,
                n_regions,
                selected,
                MaxAcc::default,
                convert,
            )?,
            extreme,
        ),
        _ => unreachable!("float_rows only handles mean/min/max"),
    };
    Ok(rows)
}

fn finalize_rows<A: Accumulator<Output = f64>>(
    accs: Vec<Vec<A>>,
    to_value: impl Fn(f64) -> StatValue,
) -> Vec<Vec<StatValue>> {
    accs.into_iter()
        .map(|row| row.into_iter().map(|a| to_value(a.finalize())).collect())
        .collect()
}

fn count_accs<T: Numeric>(
    track: &Track,
    batches: &[Batch],
    n_regions: usize,
    selected: &[usize],
    init: impl Fn() -> CountAcc + Sync,
    convert: fn(T) -> i64,
) -> Result<Vec<Vec<CountAcc>>> {
    accumulate::<T, CountAcc, _, _>(track, batches, n_regions, selected, init, convert)
}

fn hist_table(accs: Vec<Vec<CountAcc>>) -> HistTable {
    let mut per_sample: Vec<CountAcc> = Vec::new();
    for region_accs in accs {
        if per_sample.is_empty() {
            per_sample = region_accs;
        } else {
            for (slot, acc) in per_sample.iter_mut().zip(region_accs) {
                slot.merge(acc);
            }
        }
    }
    let mut values: Vec<i64> = per_sample
        .iter()
        .flat_map(|acc| acc.observed().into_iter().map(|(v, _)| v))
        .collect();
    values.sort_unstable();
    values.dedup();
    let counts = values
        .iter()
        .map(|&v| per_sample.iter().map(|acc| acc.count_of(v)).collect())
        .collect();
    HistTable { values, counts }
}

/// Per-batch partial: accumulator rows keyed by region position.
type RegionAccs<A> = Vec<(usize, Vec<A>)>;

/// One partial accumulator row per region per batch, merged in region order.
fn accumulate<T, A, I, C>(
    track: &Track,
    batches: &[Batch],
    n_regions: usize,
    selected: &[usize],
    init: I,
    convert: C,
) -> Result<Vec<Vec<A>>>
where
    T: Numeric,
    A: Accumulator,
    I: Fn() -> A + Sync,
    C: Fn(T) -> A::Value + Sync,
{
    let rank1 = track.rank() == 1;
    let n_out = if rank1 { 1 } else { selected.len() };
    let n_cols = track.columns_count()?;
    let window = rayon::current_num_threads().saturating_mul(2).max(1);
    let mut out: Vec<Vec<A>> = (0..n_regions)
        .map(|_| (0..n_out).map(|_| init()).collect())
        .collect();
    for group in batches.chunks(window) {
        let partials: Vec<Result<RegionAccs<A>>> = group
            .par_iter()
            .map(|batch| {
                let raw = track
                    .read_flat::<T>(batch.span.clone())?
                    .into_raw_vec_and_offset()
                    .0;
                let mut per_region: BTreeMap<usize, Vec<A>> = BTreeMap::new();
                for item in &batch.items {
                    let accs = per_region
                        .entry(item.region_pos)
                        .or_insert_with(|| (0..n_out).map(|_| init()).collect());
                    let first = (item.lo - batch.span.start) as usize;
                    let rows = (item.hi - item.lo) as usize;
                    for row in first..first + rows {
                        if rank1 {
                            accs[0].update(convert(raw[row]));
                        } else {
                            for (k, &col) in selected.iter().enumerate() {
                                accs[k].update(convert(raw[row * n_cols + col]));
                            }
                        }
                    }
                }
                Ok(per_region.into_iter().collect())
            })
            .collect();
        for batch in partials {
            for (region_pos, accs) in batch? {
                for (slot, acc) in out[region_pos].iter_mut().zip(accs) {
                    slot.merge(acc);
                }
            }
        }
    }
    Ok(out)
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

    use crate::genome::{Contig, Genome, Region};
    use crate::io::Dtype;
    use crate::store::PbzStore;
    use crate::track::TrackConfig;

    fn fixture(dir: &std::path::Path) -> PbzStore {
        let genome = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 10,
            },
            Contig {
                name: "chr2".into(),
                length: 6,
            },
        ])
        .unwrap();
        let mut store = PbzStore::create(dir.join("stat.pbz")).unwrap();
        store
            .create_track("depth", genome.clone(), TrackConfig::new(Dtype::I32))
            .unwrap();
        store
            .create_track(
                "af",
                genome.clone(),
                TrackConfig::new(Dtype::F32)
                    .columns(vec!["s1".into(), "s2".into()])
                    .column_dim("sample"),
            )
            .unwrap();
        let depth = store.track("depth").unwrap();
        let chr1 = genome.resolve(&"chr1".parse().unwrap()).unwrap();
        depth
            .write_region(
                &chr1,
                ndarray::Array1::from(vec![5i32, 5, 5, 7, 7, 0, 0, 0, 0, 0]).into_dyn(),
            )
            .unwrap();
        let af = store.track("af").unwrap();
        let window = genome.resolve(&"chr1:2-6".parse().unwrap()).unwrap();
        af.write_region(
            &window,
            ndarray::Array2::from_shape_vec(
                (4, 2),
                vec![0.5f32, 1.0, 0.5, 1.0, 0.25, 1.0, 0.25, 1.0],
            )
            .unwrap()
            .into_dyn(),
        )
        .unwrap();
        store
    }

    fn whole(track: &crate::track::Track) -> Vec<Region> {
        track
            .genome()
            .iter()
            .map(|(id, c)| Region {
                contig: id,
                start: 0,
                end: c.length,
            })
            .collect()
    }

    #[test]
    fn run_mean_median_on_integer_track() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = fixture(dir.path());
        let track = store.track("depth").unwrap();
        let regions = whole(track);

        let mean = run(track, &regions, StatKind::Mean, &StatOptions::default()).unwrap();
        assert!(mean.samples.is_empty());
        let StatResult::PerRegion(rows) = mean.result else {
            panic!("expected rows")
        };
        assert_eq!(
            rows,
            vec![vec![StatValue::Float(2.9)], vec![StatValue::Float(0.0)]]
        );

        let head = Region {
            contig: regions[0].contig,
            start: 0,
            end: 5,
        };
        let median = run(track, &[head], StatKind::Median, &StatOptions::default()).unwrap();
        let StatResult::PerRegion(rows) = median.result else {
            panic!("expected rows")
        };
        assert_eq!(rows, vec![vec![StatValue::Int(5)]]);
    }

    #[test]
    fn run_hist_unions_overlapping_regions() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = fixture(dir.path());
        let track = store.track("depth").unwrap();
        let contig = whole(track)[0].contig;
        let overlapping = [
            Region {
                contig,
                start: 0,
                end: 5,
            },
            Region {
                contig,
                start: 3,
                end: 8,
            },
        ];
        let out = run(track, &overlapping, StatKind::Hist, &StatOptions::default()).unwrap();
        let StatResult::Hist(table) = out.result else {
            panic!("expected hist")
        };
        // union 0..8 -> values 5,5,5,7,7,0,0,0
        assert_eq!(table.values, vec![0, 5, 7]);
        assert_eq!(table.counts, vec![vec![3], vec![3], vec![2]]);
    }

    #[test]
    fn run_mean_skips_nan_and_subsets_columns() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = fixture(dir.path());
        let track = store.track("af").unwrap();
        let regions = whole(track);
        let options = StatOptions {
            columns: Some(vec!["s2".into(), "s1".into()]),
        };
        let out = run(track, &regions, StatKind::Mean, &options).unwrap();
        assert_eq!(out.samples, vec!["s2".to_string(), "s1".to_string()]);
        let StatResult::PerRegion(rows) = out.result else {
            panic!("expected rows")
        };
        assert_eq!(
            rows[0],
            vec![StatValue::Float(1.0), StatValue::Float(0.375)]
        );
        let StatValue::Float(chr2_s2) = rows[1][0] else {
            panic!("expected float")
        };
        assert!(chr2_s2.is_nan());
    }

    #[test]
    fn run_rejects_median_on_float_and_bad_columns() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = fixture(dir.path());
        let af = store.track("af").unwrap();
        let regions = whole(af);
        let err = run(af, &regions, StatKind::Median, &StatOptions::default()).unwrap_err();
        assert!(err.to_string().contains("median"), "{err}");
        assert!(err.to_string().contains("f32"), "{err}");

        let options = StatOptions {
            columns: Some(vec!["nope".into()]),
        };
        let err = run(af, &regions, StatKind::Mean, &options).unwrap_err();
        assert!(err.to_string().contains("s1"), "{err}");

        let depth = store.track("depth").unwrap();
        let options = StatOptions {
            columns: Some(vec!["s1".into()]),
        };
        assert!(run(depth, &whole(depth), StatKind::Mean, &options).is_err());
    }

    #[test]
    fn run_hist_on_rank2_track_counts_per_sample() {
        let dir = tempfile::TempDir::new().unwrap();
        let genome = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 4,
        }])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("h.pbz")).unwrap();
        store
            .create_track(
                "cov",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(vec!["s1".into(), "s2".into()])
                    .column_dim("sample"),
            )
            .unwrap();
        let track = store.track("cov").unwrap();
        let region = genome.resolve(&"chr1".parse().unwrap()).unwrap();
        track
            .write_region(
                &region,
                ndarray::Array2::from_shape_vec((4, 2), vec![1i32, 5, 1, 5, 2, 5, 0, 7])
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        let out = run(track, &[region], StatKind::Hist, &StatOptions::default()).unwrap();
        assert_eq!(out.samples, vec!["s1".to_string(), "s2".to_string()]);
        let StatResult::Hist(table) = out.result else {
            panic!("expected hist")
        };
        assert_eq!(table.values, vec![0, 1, 2, 5, 7]);
        assert_eq!(
            table.counts,
            vec![vec![1, 0], vec![2, 0], vec![1, 0], vec![0, 3], vec![0, 1]]
        );
    }
}
