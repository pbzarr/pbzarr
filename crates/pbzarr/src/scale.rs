//! Multiscale pyramid: compute and publish per-track mean levels.
//!
//! `scale` bins each sequence of a track independently (bins never straddle
//! sequence boundaries; the last bin of a sequence is ragged when the factor
//! does not divide its length), writes one float32 `mean` level array per
//! factor under `<track>/scales/<factor>/mean`, then publishes the pyramid by
//! writing the `multiscales` attr (plus its `zarr_conventions` entry) on the
//! track group. Publication ordering guarantees a crash leaves a valid
//! base-only track: level arrays without the attr are invisible orphans and
//! are cleaned up by the next `scale` run.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayD};
use serde_json::{Value, json};
use zarrs::array::{Array, ArraySubset, FillValue, data_type};
use zarrs::storage::{ReadableWritableListableStorageTraits, StorePrefix};

use crate::error::PbzError;
use crate::genome::{ContigId, Region};
use crate::io::{Dtype, Numeric};
use crate::store::default_data_codecs;
use crate::track::Track;
use crate::{PbzStore, Result};

/// Minted uuid of the zarr-conventions `multiscales` convention (v0.1).
pub const MULTISCALES_CONVENTION_UUID: &str = "d35379db-88df-4056-af3a-620245f8e347";
const MULTISCALES_CONVENTION_NAME: &str = "multiscales";
const MULTISCALES_SCHEMA_URL: &str =
    "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json";
const MULTISCALES_SPEC_URL: &str =
    "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md";

/// Position-chunk length for level arrays, in bins (clamped to level length).
/// Full column width per chunk; no sharding.
const LEVEL_CHUNK_LEN: u64 = 262_144;

/// Byte cap for one level chunk (`chunk_len * n_columns * 4`). Bounds the
/// chunk length on wide cohort tracks; blosc rejects buffers near 2 GiB.
const LEVEL_CHUNK_TARGET_BYTES: u64 = 64 << 20;

/// Byte budget for one slab of base data (`slab_len * n_columns * 4`).
const SLAB_TARGET_BYTES: u64 = 64 << 20;

/// Tier A cap on the slab alignment (the lcm of the factors swept from base):
/// factor sets whose alignment exceeds this fall back to the per-factor
/// multi-pass path (Tier B).
const TIER_A_MAX_ALIGN: u64 = 1 << 22;

/// Cap on the work-unit alignment (the lcm of ALL factors). Units align to
/// that lcm so no bin of any factor spans two units; factor sets whose lcm
/// exceeds this use whole contigs as units instead (reduced parallelism,
/// same results).
const UNIT_MAX_ALIGN: u64 = 1 << 26;

/// Default ladder: start at 32, multiply by 8 while the largest sequence
/// still contributes more than 2000 bins at the current factor.
const LADDER_START: u64 = 32;
const LADDER_RATIO: u64 = 8;
const LADDER_MAX_BINS: u64 = 2000;

/// A downsampling statistic. Only `Mean` is implemented in the POC; the
/// other variants are rejected by [`ScaleConfig`] validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    /// Per-bin arithmetic mean (f64 sum + i64 count, float32 output).
    Mean,
    /// Per-bin minimum (unimplemented).
    Min,
    /// Per-bin maximum (unimplemented).
    Max,
    /// Per-bin count of non-NaN positions (unimplemented).
    ValidCount,
}

impl Stat {
    fn name(self) -> &'static str {
        match self {
            Stat::Mean => "mean",
            Stat::Min => "min",
            Stat::Max => "max",
            Stat::ValidCount => "valid_count",
        }
    }
}

/// Configuration for [`scale`].
#[derive(Debug, Clone)]
pub struct ScaleConfig {
    /// Downsampling factors: positive, unique, ascending, `>= 2`. `None`
    /// selects the default ladder (32, 256, 2048, ... by the 2000-bin rule).
    pub factors: Option<Vec<u64>>,
    /// Statistics to compute per factor. POC: must be `[Stat::Mean]`.
    pub stats: Vec<Stat>,
    /// Worker threads computing level bins (a single writer thread performs
    /// every level-array write). Clamped to at least 1.
    pub workers: usize,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            factors: None,
            stats: vec![Stat::Mean],
            workers: 4,
        }
    }
}

impl ScaleConfig {
    fn validate(&self) -> Result<()> {
        if self.stats.is_empty() {
            return Err(PbzError::Metadata("scale: no statistics requested".into()));
        }
        for stat in &self.stats {
            if *stat != Stat::Mean {
                return Err(PbzError::Metadata(format!(
                    "scale: statistic '{}' is unimplemented (POC supports mean only)",
                    stat.name()
                )));
            }
        }
        if let Some(factors) = &self.factors {
            if factors.is_empty() {
                return Err(PbzError::Metadata("scale: empty factor list".into()));
            }
            let mut prev = 1u64;
            for &f in factors {
                if f < 2 {
                    return Err(PbzError::Metadata(format!(
                        "scale: factor {f} must be >= 2"
                    )));
                }
                if f <= prev {
                    return Err(PbzError::Metadata(format!(
                        "scale: factors must be unique and ascending (found {f} after {prev})"
                    )));
                }
                prev = f;
            }
        }
        Ok(())
    }
}

/// Summary of one written level, for CLI reporting.
#[derive(Debug, Clone)]
pub struct LevelReport {
    /// The downsampling factor of this level.
    pub factor: u64,
    /// Level length: `Σ_c ceil(L_c / factor)` bins.
    pub bins: u64,
    /// Bytes stored under `scales/<factor>/` (metadata + chunks).
    pub bytes_written: u64,
}

/// Summary of a [`scale`] run, one entry per factor in ascending order.
#[derive(Debug, Clone)]
pub struct ScaleReport {
    /// Per-factor level summaries.
    pub levels: Vec<LevelReport>,
}

/// Compute and publish the multiscale pyramid for `track`.
///
/// Errors if the track already has a published pyramid (unpublish/rescale is
/// not implemented), or if its dtype is not int32/float32/bool. A leftover
/// `scales/` subtree without a published `multiscales` attr (crashed prior
/// run) is deleted and rewritten idempotently. The already-published error
/// still refreshes the store-root consolidated metadata first, so retrying
/// after a crash between publication and consolidation heals the root map.
pub fn scale(store: &PbzStore, track: &str, config: &ScaleConfig) -> Result<ScaleReport> {
    config.validate()?;
    let t = store
        .track(track)
        .ok_or_else(|| PbzError::Store(format!("scale: no track {track:?} in store")))?;
    match t.dtype() {
        Dtype::I32 | Dtype::F32 | Dtype::Bool => {}
        other => {
            return Err(PbzError::InvalidDtype {
                dtype: format!("scale: unsupported source dtype {other} (int32, float32, bool)"),
            });
        }
    }

    // Rescale is not in POC scope: a published pyramid must be unpublished
    // first, and unpublish does not exist yet.
    let group = zarrs::group::Group::open(store.storage.clone(), &format!("/{track}"))
        .map_err(|e| PbzError::Store(e.to_string()))?;
    if group.attributes().contains_key("multiscales") {
        // Recovery path: a crash between the publication write (2) and the
        // consolidation refresh (3) leaves the pyramid published on disk but
        // hidden from the root map, and this is the branch a retry lands in.
        // Refuse the rescale, but heal the map first so the retry is not a
        // no-op that leaves the store stale until some later publication.
        drop(group);
        store.consolidate_metadata()?;
        return Err(PbzError::Metadata(format!(
            "scale: track {track:?} already has a published pyramid; unpublish/rescale is not implemented"
        )));
    }
    drop(group);

    // Idempotent cleanup: any `scales/` content without the attr is an
    // invisible orphan from a crashed run; delete and rewrite.
    let scales_prefix =
        StorePrefix::new(format!("{track}/scales/")).map_err(|e| PbzError::Store(e.to_string()))?;
    store
        .storage
        .erase_prefix(&scales_prefix)
        .map_err(|e| PbzError::Store(e.to_string()))?;

    let factors = match &config.factors {
        Some(f) => f.clone(),
        None => {
            let max_len = t
                .genome()
                .contigs()
                .iter()
                .map(|c| c.length)
                .max()
                .unwrap_or(0);
            default_ladder(max_len)
        }
    };

    // (1) Write all level arrays under scales/.
    zarrs::group::GroupBuilder::new()
        .build(store.storage.clone(), &format!("/{track}/scales"))
        .map_err(|e| PbzError::Store(e.to_string()))?
        .store_metadata()
        .map_err(|e| PbzError::Store(e.to_string()))?;

    let source_fill = source_fill_as_f64(t)?;
    let nan_aware = t.dtype() == Dtype::F32 && source_fill.is_nan();
    let mut level_arrays = Vec::with_capacity(factors.len());
    for &factor in &factors {
        level_arrays.push(create_level(store, t, factor, source_fill)?);
    }

    let n_cols = t.columns_count()? as u64;
    let parents = factor_parents(&factors);
    match tier_a_align(&factors, &parents, n_cols) {
        // Tier A: one traversal of the base data feeds every root factor;
        // child factors cascade from their parent's bin accumulators.
        Some(align) => {
            compute_levels_tier_a(t, &level_arrays, &parents, align, nan_aware, config.workers)?
        }
        // Tier B: pathological factor set; per-factor multi-pass fallback.
        None => {
            for lvl in &level_arrays {
                compute_level_tier_b(t, lvl, nan_aware)?;
            }
        }
    }

    let mut levels = Vec::with_capacity(factors.len());
    for lvl in &level_arrays {
        let level_prefix = StorePrefix::new(format!("{track}/scales/{}/", lvl.factor))
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let bytes_written = store
            .storage
            .size_prefix(&level_prefix)
            .map_err(|e| PbzError::Store(e.to_string()))?;
        levels.push(LevelReport {
            factor: lvl.factor,
            bins: lvl.bins,
            bytes_written,
        });
    }

    // (2) The publication write: multiscales attr + its conventions entry,
    // in one track zarr.json write. Base attrs are preserved.
    let mut group = zarrs::group::Group::open(store.storage.clone(), &format!("/{track}"))
        .map_err(|e| PbzError::Store(e.to_string()))?;
    group
        .attributes_mut()
        .insert("multiscales".to_owned(), multiscales_attr(&factors));
    let conventions = group
        .attributes_mut()
        .get_mut("zarr_conventions")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| {
            PbzError::Metadata(format!("scale: track {track:?} missing zarr_conventions"))
        })?;
    conventions.push(json!({
        "uuid": MULTISCALES_CONVENTION_UUID,
        "name": MULTISCALES_CONVENTION_NAME,
        "schema_url": MULTISCALES_SCHEMA_URL,
        "spec_url": MULTISCALES_SPEC_URL,
    }));
    group
        .store_metadata()
        .map_err(|e| PbzError::Store(e.to_string()))?;

    // The pyramid is published on disk as of the write above, so seal the
    // live handle before anything else can fail: a consolidation error must
    // not leave a published track still writable through this handle.
    t.seal();

    // (3) Refresh the store-root consolidated metadata.
    store.consolidate_metadata()?;

    Ok(ScaleReport { levels })
}

/// The default factor ladder for a genome whose largest sequence has
/// `max_len` positions. Always at least one factor.
fn default_ladder(max_len: u64) -> Vec<u64> {
    let mut factors = vec![LADDER_START];
    let mut f = LADDER_START;
    while max_len.div_ceil(f) > LADDER_MAX_BINS {
        f = f.saturating_mul(LADDER_RATIO);
        factors.push(f);
    }
    factors
}

/// The `multiscales` track-group attr: base entry first, then one entry per
/// factor in ascending order.
fn multiscales_attr(factors: &[u64]) -> Value {
    let mut layout = vec![json!({"asset": "values"})];
    for &f in factors {
        layout.push(json!({
            "asset": format!("scales/{f}/mean"),
            "derived_from": "values",
            "transform": {"perbase:ragged_axis_scale": {
                "dimension": "position",
                "factor": f,
                "anchor": "segment-start",
                "last_bin": "clip",
            }},
            "resampling_method": "average",
        }));
    }
    json!({"layout": layout})
}

/// A created (not yet published) level array and its geometry.
struct Level {
    factor: u64,
    /// Level length: `Σ_c ceil(L_c / factor)` bins.
    bins: u64,
    array: Array<dyn ReadableWritableListableStorageTraits>,
}

/// Create the `scales/<factor>/mean` array (group and metadata only; no
/// chunk data is written).
fn create_level(store: &PbzStore, t: &Track, factor: u64, source_fill: f64) -> Result<Level> {
    let track = t.name();
    let level_len: u64 = t
        .genome()
        .contigs()
        .iter()
        .map(|c| c.length.div_ceil(factor))
        .sum();
    let n_cols = t.columns_count()? as u64;

    zarrs::group::GroupBuilder::new()
        .build(store.storage.clone(), &format!("/{track}/scales/{factor}"))
        .map_err(|e| PbzError::Store(e.to_string()))?
        .store_metadata()
        .map_err(|e| PbzError::Store(e.to_string()))?;

    // Level fill = the mean of an all-fill bin: f32 of the source fill.
    let level_fill = FillValue::from(source_fill as f32);

    let chunk_len = LEVEL_CHUNK_LEN
        .min(LEVEL_CHUNK_TARGET_BYTES / (4 * n_cols.max(1)))
        .max(1)
        .min(level_len.max(1));
    let (shape, chunks, dim_names): (Vec<u64>, Vec<u64>, Vec<&str>) = if t.rank() == 2 {
        let col_dim = t.column_dim().unwrap_or("column");
        (
            vec![level_len, n_cols],
            vec![chunk_len, n_cols.max(1)],
            vec!["bin", col_dim],
        )
    } else {
        (vec![level_len], vec![chunk_len], vec!["bin"])
    };
    let mut builder =
        zarrs::array::ArrayBuilder::new(shape, chunks, data_type::float32(), level_fill);
    builder
        .dimension_names(dim_names.into())
        .bytes_to_bytes_codecs(default_data_codecs(Dtype::F32)?);
    let level = builder
        .build(
            store.storage.clone(),
            &format!("/{track}/scales/{factor}/mean"),
        )
        .map_err(|e| PbzError::Store(e.to_string()))?;
    level
        .store_metadata()
        .map_err(|e| PbzError::Store(e.to_string()))?;

    Ok(Level {
        factor,
        bins: level_len,
        array: level,
    })
}

/// For each factor (ascending), the index of its cascade parent: the largest
/// earlier factor that divides it. `None` marks a root, which accumulates
/// directly from base values; children aggregate their parent's bins.
fn factor_parents(factors: &[u64]) -> Vec<Option<usize>> {
    factors
        .iter()
        .enumerate()
        .map(|(i, &f)| (0..i).rev().find(|&j| f % factors[j] == 0))
        .collect()
}

/// Slab alignment (the lcm of the ROOT factors) for the Tier A single-pass
/// path, or `None` when the set is pathological and the per-factor
/// multi-pass fallback (Tier B) must run instead: the root lcm exceeds
/// [`TIER_A_MAX_ALIGN`], or one aligned slab unit at full column width
/// overflows the slab byte budget (very wide cohorts at a large root lcm).
fn tier_a_align(factors: &[u64], parents: &[Option<usize>], n_cols: u64) -> Option<u64> {
    let mut align = 1u64;
    for (i, &f) in factors.iter().enumerate() {
        if parents[i].is_some() {
            continue; // children never sweep base; they don't constrain slabs
        }
        align = (align / gcd(align, f)).checked_mul(f)?;
        if align > TIER_A_MAX_ALIGN {
            return None;
        }
    }
    if align.checked_mul(4 * n_cols.max(1))? > SLAB_TARGET_BYTES {
        return None;
    }
    Some(align)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Tier A dispatch on the source dtype.
fn compute_levels_tier_a(
    t: &Track,
    levels: &[Level],
    parents: &[Option<usize>],
    align: u64,
    nan_aware: bool,
    workers: usize,
) -> Result<()> {
    match t.dtype() {
        Dtype::I32 => {
            let ctx = PassCtx::<i32>::new(t, levels, parents, align, false, |v| v as f64)?;
            run_units(&ctx, workers)
        }
        Dtype::F32 => {
            let ctx = PassCtx::<f32>::new(t, levels, parents, align, nan_aware, |v| v as f64)?;
            run_units(&ctx, workers)
        }
        Dtype::Bool => {
            let ctx =
                PassCtx::<bool>::new(
                    t,
                    levels,
                    parents,
                    align,
                    false,
                    |v| {
                        if v { 1.0 } else { 0.0 }
                    },
                )?;
            run_units(&ctx, workers)
        }
        other => Err(PbzError::InvalidDtype {
            dtype: format!("scale: unsupported source dtype {other}"),
        }),
    }
}

/// Tier B dispatch on the source dtype (per-factor multi-pass fallback).
fn compute_level_tier_b(t: &Track, lvl: &Level, nan_aware: bool) -> Result<()> {
    match t.dtype() {
        Dtype::I32 => compute_level::<i32>(t, &lvl.array, lvl.factor, false, |v| v as f64),
        Dtype::F32 => compute_level::<f32>(t, &lvl.array, lvl.factor, nan_aware, |v| v as f64),
        Dtype::Bool => {
            compute_level::<bool>(
                t,
                &lvl.array,
                lvl.factor,
                false,
                |v| if v { 1.0 } else { 0.0 },
            )
        }
        other => Err(PbzError::InvalidDtype {
            dtype: format!("scale: unsupported source dtype {other}"),
        }),
    }
}

/// Shared inputs of one Tier A pass, borrowed by every worker thread.
struct PassCtx<'a, T> {
    t: &'a Track,
    levels: &'a [Level],
    parents: &'a [Option<usize>],
    /// Slab alignment: the lcm of the ROOT factors (from [`tier_a_align`]).
    align: u64,
    /// Byte budget for one slab of base data ([`SLAB_TARGET_BYTES`] outside
    /// tests).
    slab_target_bytes: u64,
    nan_aware: bool,
    to_f64: fn(T) -> f64,
    rank2: bool,
    n_cols: usize,
}

impl<'a, T: Numeric> PassCtx<'a, T> {
    fn new(
        t: &'a Track,
        levels: &'a [Level],
        parents: &'a [Option<usize>],
        align: u64,
        nan_aware: bool,
        to_f64: fn(T) -> f64,
    ) -> Result<Self> {
        Ok(Self {
            t,
            levels,
            parents,
            align,
            slab_target_bytes: SLAB_TARGET_BYTES,
            nan_aware,
            to_f64,
            rank2: t.rank() == 2,
            n_cols: t.columns_count()?,
        })
    }
}

/// One parallel work unit: a contig-local position range. Boundaries land on
/// multiples of lcm(all factors) (or the contig ends), so no factor's bin
/// spans two units and the child-bin carry stays unit-internal; units share
/// no state and may run in any order.
struct Unit {
    cid: ContigId,
    start: u64,
    end: u64,
    /// Per level: the absolute bin index of this contig's FIRST bin (the
    /// same for every unit of the contig).
    level_bases: Vec<u64>,
}

/// A block of completed bins for one level, sent worker -> writer.
struct LevelBlock {
    level: usize,
    bin0: u64,
    means: Vec<f32>,
}

/// Split the genome into contig-local work units. Unit length targets one
/// slab budget of base data rounded to the unit alignment, so each unit is
/// roughly one read; when lcm(all factors) exceeds [`UNIT_MAX_ALIGN`] (or
/// overflows), whole contigs become units instead: correct, just less
/// parallel.
fn build_units<T: Numeric>(ctx: &PassCtx<'_, T>) -> Vec<Unit> {
    let unit_align = ctx.levels.iter().try_fold(1u64, |a, l| {
        let m = (a / gcd(a, l.factor)).checked_mul(l.factor)?;
        (m <= UNIT_MAX_ALIGN).then_some(m)
    });
    let per_pos_bytes = 4 * ctx.n_cols.max(1) as u64;
    let slab_positions = (ctx.slab_target_bytes / per_pos_bytes).max(1);

    let mut units = Vec::new();
    let mut level_bases = vec![0u64; ctx.levels.len()];
    for (cid, contig) in ctx.t.genome().iter() {
        let len = contig.length;
        if len > 0 {
            let unit_len = match unit_align {
                Some(a) => (slab_positions / a).max(1) * a,
                None => len,
            };
            let mut start = 0u64;
            while start < len {
                let end = (start + unit_len).min(len);
                units.push(Unit {
                    cid,
                    start,
                    end,
                    level_bases: level_bases.clone(),
                });
                start = end;
            }
        }
        for (li, lvl) in ctx.levels.iter().enumerate() {
            level_bases[li] += len.div_ceil(lvl.factor);
        }
    }
    units
}

/// Run the Tier A pass with `workers` compute threads and the calling
/// thread as the single writer.
///
/// Workers claim units off a shared index and send completed-bin blocks
/// over a bounded channel; only the writer touches the level arrays, so no
/// concurrent read-modify-write exists on shared level chunks whatever the
/// chunk geometry. The first worker or writer error stops the run (workers
/// check the flag between units; the writer keeps draining so no worker
/// blocks on a full channel) and is returned. `workers = 1` runs the exact
/// same machinery and produces identical values.
fn run_units<T: Numeric>(ctx: &PassCtx<'_, T>, workers: usize) -> Result<()> {
    let units = build_units(ctx);
    let units: &[Unit] = &units;
    let workers = workers.clamp(1, units.len().max(1));
    // Backpressure: a few blocks in flight per worker keeps peak memory at
    // ~ workers x (slab + accumulators) + cap x one block.
    let cap = workers * 2;
    let (tx, rx) = std::sync::mpsc::sync_channel::<LevelBlock>(cap);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_err: Mutex<Option<PbzError>> = Mutex::new(None);
    let record = |e: PbzError| {
        failed.store(true, Ordering::Release);
        let mut slot = first_err.lock().expect("scale: error slot poisoned");
        if slot.is_none() {
            *slot = Some(e);
        }
    };

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let tx = tx.clone();
            let (next, failed, record) = (&next, &failed, &record);
            scope.spawn(move || {
                loop {
                    if failed.load(Ordering::Acquire) {
                        break;
                    }
                    let Some(unit) = units.get(next.fetch_add(1, Ordering::Relaxed)) else {
                        break;
                    };
                    let mut emit = |level: usize, bin0: u64, means: Vec<f32>| {
                        tx.send(LevelBlock { level, bin0, means })
                            .map_err(|_| PbzError::Store("scale: writer thread hung up".into()))
                    };
                    if let Err(e) = compute_unit(ctx, unit, &mut emit) {
                        record(e);
                        break;
                    }
                }
            });
        }
        drop(tx); // the writer's recv loop ends when the last worker exits

        // Single writer: every level-array write happens on this thread.
        while let Ok(block) = rx.recv() {
            if failed.load(Ordering::Acquire) {
                continue; // drain so workers never block on a full channel
            }
            let arr = &ctx.levels[block.level].array;
            if let Err(e) = write_f32_bins(arr, ctx.rank2, block.bin0, ctx.n_cols, block.means) {
                record(e);
            }
        }
    });

    match first_err.into_inner().expect("scale: error slot poisoned") {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Run the single-pass + cascade core over one unit, emitting every
/// factor's completed bins as `emit(level index, absolute first bin, f32
/// means)`.
///
/// Slabs start on multiples of `ctx.align` (unit starts are multiples of
/// lcm(all factors), hence of the root lcm), so every root's bins complete
/// within the slab window; the final slab clips at the unit end, which is
/// either a multiple of every factor or the sequence end (where the clip
/// rule completes all bins). Children aggregate their parent's completed
/// f64 sums and i64 counts (never the rounded f32 outputs) at bin
/// granularity; a child bin that straddles slabs carries exactly one
/// partial (sum, count) row across slabs, flushed by the unit end.
fn compute_unit<T: Numeric>(
    ctx: &PassCtx<'_, T>,
    unit: &Unit,
    emit: &mut dyn FnMut(usize, u64, Vec<f32>) -> Result<()>,
) -> Result<()> {
    let (levels, parents, n_cols) = (ctx.levels, ctx.parents, ctx.n_cols);
    let per_pos_bytes = 4 * n_cols.max(1) as u64;
    // Tier A guarantees one aligned unit fits the budget.
    let slab = ((ctx.slab_target_bytes / per_pos_bytes) / ctx.align).max(1) * ctx.align;

    // Per-child partial bin carried across slabs: the one child bin the
    // previous slabs left incomplete (sum and count per column).
    let mut carry_sums = vec![vec![0f64; n_cols]; levels.len()];
    let mut carry_counts = vec![vec![0i64; n_cols]; levels.len()];
    let mut slab_start = unit.start;
    while slab_start < unit.end {
        let slab_end = (slab_start + slab).min(unit.end);
        let at_end = slab_end == unit.end;
        let rows = (slab_end - slab_start) as usize;
        let region = Region {
            contig: unit.cid,
            start: slab_start,
            end: slab_end,
        };
        let data: ArrayD<T> = ctx.t.read_region(&region)?;
        let data = data
            .into_shape_with_order((rows, n_cols))
            .map_err(|e| PbzError::Store(format!("scale: reshape slab: {e}")))?;

        // Per factor, the bins COMPLETED this slab: (first bin index,
        // contig-local; sums; counts), consumed by child factors below.
        let mut done: Vec<(u64, Vec<f64>, Vec<i64>)> = Vec::with_capacity(levels.len());
        for (li, lvl) in levels.iter().enumerate() {
            let factor = lvl.factor;
            let (start_bin, sums, counts) = match parents[li] {
                // Root: sweep the slab. slab_start is a multiple of
                // `align` and hence of `factor`, so every bin completes.
                None => {
                    let n_bins = (slab_end - slab_start).div_ceil(factor) as usize;
                    let mut sums = vec![0f64; n_bins * n_cols];
                    let mut counts = vec![0i64; n_bins * n_cols];
                    for (i, row) in data.outer_iter().enumerate() {
                        let bin = i / factor as usize;
                        for (j, &v) in row.iter().enumerate() {
                            let v = (ctx.to_f64)(v);
                            if ctx.nan_aware && v.is_nan() {
                                continue;
                            }
                            sums[bin * n_cols + j] += v;
                            counts[bin * n_cols + j] += 1;
                        }
                    }
                    (slab_start / factor, sums, counts)
                }
                // Child: aggregate the parent's completed bins. Child bin
                // j is exactly the union of parent bins j*k..(j+1)*k
                // (clipped at the sequence end).
                Some(pi) => {
                    let (p_start, p_sums, p_counts) = &done[pi];
                    let p_start = *p_start;
                    let k = factor / levels[pi].factor;
                    let p_end = p_start + (p_counts.len() / n_cols.max(1)) as u64;
                    let j0 = p_start / k;
                    let j1 = p_end.div_ceil(k);
                    let complete = if at_end { j1 } else { p_end / k };
                    let n_bins = (j1 - j0) as usize;
                    let mut sums = vec![0f64; n_bins * n_cols];
                    let mut counts = vec![0i64; n_bins * n_cols];
                    // Seed the partial bin carried from previous slabs.
                    if p_start % k != 0 {
                        sums[..n_cols].copy_from_slice(&carry_sums[li]);
                        counts[..n_cols].copy_from_slice(&carry_counts[li]);
                    }
                    for pb in p_start..p_end {
                        let dst = (pb / k - j0) as usize * n_cols;
                        let src = (pb - p_start) as usize * n_cols;
                        for j in 0..n_cols {
                            sums[dst + j] += p_sums[src + j];
                            counts[dst + j] += p_counts[src + j];
                        }
                    }
                    // The slab may leave the last buffered bin
                    // incomplete: park it in the carry and emit the rest.
                    if complete < j1 {
                        let last = (n_bins - 1) * n_cols;
                        carry_sums[li].copy_from_slice(&sums[last..last + n_cols]);
                        carry_counts[li].copy_from_slice(&counts[last..last + n_cols]);
                        sums.truncate((complete - j0) as usize * n_cols);
                        counts.truncate((complete - j0) as usize * n_cols);
                    }
                    (j0, sums, counts)
                }
            };
            emit(
                li,
                unit.level_bases[li] + start_bin,
                bin_means(&sums, &counts),
            )?;
            done.push((start_bin, sums, counts));
        }
        slab_start = slab_end;
    }
    Ok(())
}

/// Per-bin f32 means from f64 sums and i64 counts (NaN for empty bins).
fn bin_means(sums: &[f64], counts: &[i64]) -> Vec<f32> {
    sums.iter()
        .zip(counts)
        .map(|(&s, &c)| {
            if c == 0 {
                f32::NAN
            } else {
                (s / c as f64) as f32
            }
        })
        .collect()
}

/// Write `means` (`n_bins x n_cols`, row-major) at level bins
/// `[bin0, bin0 + n_bins)`, full column width. Empty ranges are a no-op.
fn write_f32_bins(
    level: &Array<dyn ReadableWritableListableStorageTraits>,
    rank2: bool,
    bin0: u64,
    n_cols: usize,
    means: Vec<f32>,
) -> Result<()> {
    let n_bins = means.len() / n_cols.max(1);
    if n_bins == 0 {
        return Ok(());
    }
    let out: ArrayD<f32> = if rank2 {
        Array2::from_shape_vec((n_bins, n_cols), means)
            .map_err(|e| PbzError::Store(format!("scale: shape level slab: {e}")))?
            .into_dyn()
    } else {
        Array1::from(means).into_dyn()
    };
    #[allow(clippy::single_range_in_vec_init)]
    let subset = if rank2 {
        ArraySubset::new_with_ranges(&[bin0..bin0 + n_bins as u64, 0..n_cols as u64])
    } else {
        ArraySubset::new_with_ranges(&[bin0..bin0 + n_bins as u64])
    };
    level
        .store_array_subset(&subset, out)
        .map_err(|e| PbzError::Store(format!("scale: write level: {e}")))
}

/// The base `values` fill value as f64 (NaN for NaN-fill float tracks).
fn source_fill_as_f64(t: &Track) -> Result<f64> {
    let values = t.values_array()?;
    let bytes = values.fill_value().as_ne_bytes();
    let bad = || PbzError::Metadata(format!("track {:?}: unexpected fill-value size", t.name()));
    Ok(match t.dtype() {
        Dtype::I32 => f64::from(i32::from_ne_bytes(bytes.try_into().map_err(|_| bad())?)),
        Dtype::F32 => f64::from(f32::from_ne_bytes(bytes.try_into().map_err(|_| bad())?)),
        Dtype::Bool => f64::from(u8::from(*bytes.first().ok_or_else(bad)? != 0)),
        other => {
            return Err(PbzError::InvalidDtype {
                dtype: format!("scale: unsupported source dtype {other}"),
            });
        }
    })
}

/// Slab length in base positions: a multiple of `factor`, sized so
/// `slab_len * n_cols * 4` stays under [`SLAB_TARGET_BYTES`] (always at
/// least one whole bin).
fn slab_len(factor: u64, n_cols: u64) -> u64 {
    let per_pos_bytes = 4 * n_cols.max(1);
    let max_positions = (SLAB_TARGET_BYTES / per_pos_bytes).max(1);
    (max_positions / factor).max(1) * factor
}

/// Tier B fallback: fill one level array with per-bin means, sequence by
/// sequence, slab by slab (one full base traversal per factor).
///
/// Slabs start on bin boundaries and each bin is contained in exactly
/// one slab, so nothing carries across slabs along the position axis. When a
/// single full-width bin would exceed the slab budget (wide cohorts at large
/// factors), the column axis is slabbed instead: each read covers whole bins
/// over a column subrange, and nothing carries across column blocks either.
fn compute_level<T: Numeric>(
    t: &Track,
    level: &Array<dyn ReadableWritableListableStorageTraits>,
    factor: u64,
    nan_aware: bool,
    to_f64: impl Fn(T) -> f64,
) -> Result<()> {
    let rank2 = t.rank() == 2;
    let n_cols = t.columns_count()?;

    // Widest column range for which one whole bin stays under the budget.
    // When it covers all columns (1-D and narrow tracks), read full width
    // through `read_region`; otherwise loop column blocks.
    let col_block = ((SLAB_TARGET_BYTES / (4 * factor)).max(1) as usize).min(n_cols);
    let full_width = !rank2 || col_block >= n_cols;
    let read_width = if full_width { n_cols } else { col_block };
    let slab = slab_len(factor, read_width as u64);

    let genome = Arc::clone(t.genome());
    let offsets = genome.offsets();
    let mut level_base = 0u64;
    for (cid, contig) in genome.iter() {
        let len = contig.length;
        let flat_base = offsets[cid.as_usize()] as u64;
        let mut slab_start = 0u64;
        while slab_start < len {
            let slab_end = (slab_start + slab).min(len);
            let rows = (slab_end - slab_start) as usize;
            let n_bins = (slab_end - slab_start).div_ceil(factor) as usize;
            let bin0 = level_base + slab_start / factor;

            let mut c0 = 0usize;
            while c0 < n_cols {
                let c1 = (c0 + read_width).min(n_cols);
                let width = c1 - c0;
                let data: ArrayD<T> = if full_width {
                    let region = Region {
                        contig: cid,
                        start: slab_start,
                        end: slab_end,
                    };
                    t.read_region(&region)?
                } else {
                    t.read_flat_columns(
                        flat_base + slab_start..flat_base + slab_end,
                        c0 as u64..c1 as u64,
                    )?
                };
                let data = data
                    .into_shape_with_order((rows, width))
                    .map_err(|e| PbzError::Store(format!("scale: reshape slab: {e}")))?;

                let mut sums = vec![0f64; n_bins * width];
                let mut counts = vec![0i64; n_bins * width];
                for (i, row) in data.outer_iter().enumerate() {
                    let bin = i / factor as usize;
                    for (j, &v) in row.iter().enumerate() {
                        let v = to_f64(v);
                        if nan_aware && v.is_nan() {
                            continue;
                        }
                        sums[bin * width + j] += v;
                        counts[bin * width + j] += 1;
                    }
                }
                let means: Vec<f32> = sums
                    .iter()
                    .zip(&counts)
                    .map(|(&s, &c)| {
                        if c == 0 {
                            f32::NAN
                        } else {
                            (s / c as f64) as f32
                        }
                    })
                    .collect();

                let out: ArrayD<f32> = if rank2 {
                    Array2::from_shape_vec((n_bins, width), means)
                        .map_err(|e| PbzError::Store(format!("scale: shape level slab: {e}")))?
                        .into_dyn()
                } else {
                    Array1::from(means).into_dyn()
                };
                #[allow(clippy::single_range_in_vec_init)]
                let subset = if rank2 {
                    ArraySubset::new_with_ranges(&[
                        bin0..bin0 + n_bins as u64,
                        c0 as u64..c1 as u64,
                    ])
                } else {
                    ArraySubset::new_with_ranges(&[bin0..bin0 + n_bins as u64])
                };
                level
                    .store_array_subset(&subset, out)
                    .map_err(|e| PbzError::Store(format!("scale: write level: {e}")))?;
                c0 = c1;
            }

            slab_start = slab_end;
        }
        level_base += len.div_ceil(factor);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Contig, Genome};
    use crate::track::TrackConfig;

    /// Multi-slab cascade: a tiny slab budget forces slabs of one aligned
    /// unit (lcm(4, 6) = 12 positions), so child factor 72's bins span six
    /// slabs each and exercise every carry branch: park-only (first slab of
    /// a child bin), seed-then-park-again (middle slabs), seed-then-emit
    /// (bin-completing slab), a mid-contig complete emission (chr1 bin 0 at
    /// position 72), and the ragged sequence-end flush on both contigs.
    #[test]
    fn cascade_carry_across_small_slabs_matches_naive() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = PbzStore::create(dir.path().join("s.pbz")).unwrap();
        let genome = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 149,
            },
            Contig {
                name: "chr2".into(),
                length: 13,
            },
        ])
        .unwrap();
        store
            .create_track("depth", genome, TrackConfig::new(Dtype::I32))
            .unwrap();
        let t = store.track("depth").unwrap();
        let chr1: Vec<i32> = (0..149).map(|i| (i * 7) % 23 - 5).collect();
        let chr2: Vec<i32> = (0..13).map(|i| 100 - 9 * i).collect();
        for (name, vals) in [("chr1", &chr1), ("chr2", &chr2)] {
            let region = t
                .genome()
                .resolve(&format!("{name}:0-{}", vals.len()).parse().unwrap())
                .unwrap();
            t.write_region(&region, ndarray::Array1::from(vals.clone()).into_dyn())
                .unwrap();
        }

        // chr1 (149 = 2*72 + 5) yields child bins [0, 72), [72, 144), and a
        // ragged [144, 149); chr2 (13) is shorter than one child bin. 149
        // and 13 are divisible by none of the factors.
        let factors = [4u64, 6, 72];
        let source_fill = source_fill_as_f64(t).unwrap();
        let levels: Vec<Level> = factors
            .iter()
            .map(|&f| create_level(&store, t, f, source_fill).unwrap())
            .collect();
        let parents = factor_parents(&factors);
        assert_eq!(parents, vec![None, None, Some(1)]);
        // 48-byte budget -> slab = one aligned unit = 12 positions, so a
        // 72-position child bin spans six slabs: the middle four each seed
        // from the carry AND park back into it in the same slab. Units align
        // to lcm(4, 6, 72) = 72, so chr1 splits into units [0, 72), [72,
        // 144), [144, 149) and every carry flushes inside its unit.
        let mut ctx = PassCtx::<i32>::new(t, &levels, &parents, 12, false, |v| v as f64).unwrap();
        ctx.slab_target_bytes = 48;
        run_units(&ctx, 1).unwrap();

        for (fi, &factor) in factors.iter().enumerate() {
            let lvl = &levels[fi];
            #[allow(clippy::single_range_in_vec_init)]
            let subset = ArraySubset::new_with_ranges(&[0..lvl.bins]);
            let got = lvl
                .array
                .retrieve_array_subset::<ArrayD<f32>>(&subset)
                .unwrap()
                .into_raw_vec_and_offset()
                .0;
            let mut expect = Vec::new();
            for vals in [&chr1, &chr2] {
                for chunk in vals.chunks(factor as usize) {
                    let s: f64 = chunk.iter().map(|&v| f64::from(v)).sum();
                    expect.push((s / chunk.len() as f64) as f32);
                }
            }
            assert_eq!(got, expect, "factor {factor}");
        }
    }

    /// The unit queue under contention: the same 48-byte budget makes
    /// build_units split chr1 (149) into units [0, 72), [72, 144),
    /// [144, 149) plus one for chr2 (13), so 3 workers race 4 units through
    /// the shared index and the bounded channel, and units start mid-contig
    /// (level_bases + start_bin offsets) as well as at ragged ends.
    #[test]
    fn worker_pool_over_multiple_units_matches_naive() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = PbzStore::create(dir.path().join("s.pbz")).unwrap();
        let genome = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 149,
            },
            Contig {
                name: "chr2".into(),
                length: 13,
            },
        ])
        .unwrap();
        store
            .create_track("depth", genome, TrackConfig::new(Dtype::I32))
            .unwrap();
        let t = store.track("depth").unwrap();
        let chr1: Vec<i32> = (0..149).map(|i| (i * 11) % 31 - 7).collect();
        let chr2: Vec<i32> = (0..13).map(|i| 50 - 3 * i).collect();
        for (name, vals) in [("chr1", &chr1), ("chr2", &chr2)] {
            let region = t
                .genome()
                .resolve(&format!("{name}:0-{}", vals.len()).parse().unwrap())
                .unwrap();
            t.write_region(&region, ndarray::Array1::from(vals.clone()).into_dyn())
                .unwrap();
        }

        let factors = [4u64, 6, 72];
        let source_fill = source_fill_as_f64(t).unwrap();
        let levels: Vec<Level> = factors
            .iter()
            .map(|&f| create_level(&store, t, f, source_fill).unwrap())
            .collect();
        let parents = factor_parents(&factors);
        let mut ctx = PassCtx::<i32>::new(t, &levels, &parents, 12, false, |v| v as f64).unwrap();
        ctx.slab_target_bytes = 48;
        assert_eq!(build_units(&ctx).len(), 4);
        run_units(&ctx, 3).unwrap();

        for (fi, &factor) in factors.iter().enumerate() {
            let lvl = &levels[fi];
            #[allow(clippy::single_range_in_vec_init)]
            let subset = ArraySubset::new_with_ranges(&[0..lvl.bins]);
            let got = lvl
                .array
                .retrieve_array_subset::<ArrayD<f32>>(&subset)
                .unwrap()
                .into_raw_vec_and_offset()
                .0;
            let mut expect = Vec::new();
            for vals in [&chr1, &chr2] {
                for chunk in vals.chunks(factor as usize) {
                    let s: f64 = chunk.iter().map(|&v| f64::from(v)).sum();
                    expect.push((s / chunk.len() as f64) as f32);
                }
            }
            assert_eq!(got, expect, "factor {factor}");
        }
    }

    /// The whole-contig unit fallback: {2, 36, 2^27} is Tier A (root lcm =
    /// 2, only 2 sweeps base) but lcm(all) = 9 * 2^27 exceeds the 2^26 unit
    /// alignment cap, so build_units must emit exactly one unit per contig.
    /// A 48-byte budget makes slabs 12 positions, so inside chr1's single
    /// unit a factor-36 child bin spans three slabs (carry seeded, parked,
    /// and completed mid-contig) and the 2^27 child's lone ragged bin
    /// carries across all thirteen slabs to the contig-end flush; two
    /// workers race the two units.
    #[test]
    fn whole_contig_units_carry_across_slabs_matches_naive() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut store = PbzStore::create(dir.path().join("s.pbz")).unwrap();
        let genome = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 149,
            },
            Contig {
                name: "chr2".into(),
                length: 13,
            },
        ])
        .unwrap();
        store
            .create_track("depth", genome, TrackConfig::new(Dtype::I32))
            .unwrap();
        let t = store.track("depth").unwrap();
        let chr1: Vec<i32> = (0..149).map(|i| (i * 13) % 29 - 6).collect();
        let chr2: Vec<i32> = (0..13).map(|i| 40 - 5 * i).collect();
        for (name, vals) in [("chr1", &chr1), ("chr2", &chr2)] {
            let region = t
                .genome()
                .resolve(&format!("{name}:0-{}", vals.len()).parse().unwrap())
                .unwrap();
            t.write_region(&region, ndarray::Array1::from(vals.clone()).into_dyn())
                .unwrap();
        }

        let factors = [2u64, 36, 1 << 27];
        let parents = factor_parents(&factors);
        assert_eq!(parents, vec![None, Some(0), Some(0)]);
        // Tier A applies (the divergence under test): the root lcm is tiny
        // even though lcm(all factors) blows the unit alignment cap.
        assert_eq!(tier_a_align(&factors, &parents, 1), Some(2));

        let source_fill = source_fill_as_f64(t).unwrap();
        let levels: Vec<Level> = factors
            .iter()
            .map(|&f| create_level(&store, t, f, source_fill).unwrap())
            .collect();
        let mut ctx = PassCtx::<i32>::new(t, &levels, &parents, 2, false, |v| v as f64).unwrap();
        ctx.slab_target_bytes = 48;

        // Pin the fallback branch: exactly one unit per contig, whole span.
        let units = build_units(&ctx);
        assert_eq!(units.len(), 2);
        assert_eq!((units[0].start, units[0].end), (0, 149));
        assert_eq!((units[1].start, units[1].end), (0, 13));

        run_units(&ctx, 2).unwrap();

        for (fi, &factor) in factors.iter().enumerate() {
            let lvl = &levels[fi];
            #[allow(clippy::single_range_in_vec_init)]
            let subset = ArraySubset::new_with_ranges(&[0..lvl.bins]);
            let got = lvl
                .array
                .retrieve_array_subset::<ArrayD<f32>>(&subset)
                .unwrap()
                .into_raw_vec_and_offset()
                .0;
            let mut expect = Vec::new();
            for vals in [&chr1, &chr2] {
                for chunk in vals.chunks(factor as usize) {
                    let s: f64 = chunk.iter().map(|&v| f64::from(v)).sum();
                    expect.push((s / chunk.len() as f64) as f32);
                }
            }
            assert_eq!(got, expect, "factor {factor}");
        }
    }

    /// The Tier A/B decision itself: slab alignment is the lcm of the ROOT
    /// factors, a huge root lcm falls back to Tier B, and so does a factor
    /// set whose one aligned unit at full column width overflows the slab
    /// byte budget.
    #[test]
    fn tier_decision_alignment_cap_and_budget_guard() {
        // {4, 6, 24}: roots {4, 6} -> alignment lcm(4, 6) = 12; the child
        // 24 does not constrain slabs.
        let factors = [4u64, 6, 24];
        let parents = factor_parents(&factors);
        assert_eq!(parents, vec![None, None, Some(1)]);
        assert_eq!(tier_a_align(&factors, &parents, 1), Some(12));

        // Two large coprime roots: lcm ~ 2^42 exceeds the 2^22 cap.
        let factors = [1u64 << 21, (1 << 21) + 1];
        let parents = factor_parents(&factors);
        assert_eq!(parents, vec![None, None]);
        assert_eq!(tier_a_align(&factors, &parents, 1), None);

        // Budget guard: alignment 2^22 fits at 4 columns (2^22 * 4 * 4 B =
        // 64 MiB, exactly the budget) but not at 8 (128 MiB).
        let factors = [1u64 << 22];
        let parents = factor_parents(&factors);
        assert_eq!(tier_a_align(&factors, &parents, 1), Some(1 << 22));
        assert_eq!(tier_a_align(&factors, &parents, 4), Some(1 << 22));
        assert_eq!(tier_a_align(&factors, &parents, 8), None);
    }
}
