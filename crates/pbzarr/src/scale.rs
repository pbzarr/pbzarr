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

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD};
use serde_json::{Value, json};
use zarrs::array::{Array, ArraySubset, FillValue, data_type};
use zarrs::storage::{ReadableWritableListableStorageTraits, StorePrefix};

use crate::error::PbzError;
use crate::genome::Region;
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

/// Byte budget for one slab of base data (`slab_len * n_columns * 4`).
const SLAB_TARGET_BYTES: u64 = 64 << 20;

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
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            factors: None,
            stats: vec![Stat::Mean],
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

    let mut levels = Vec::with_capacity(factors.len());
    for &factor in &factors {
        let bins = write_level(store, t, factor)?;
        let level_prefix = StorePrefix::new(format!("{track}/scales/{factor}/"))
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let bytes_written = store
            .storage
            .size_prefix(&level_prefix)
            .map_err(|e| PbzError::Store(e.to_string()))?;
        levels.push(LevelReport {
            factor,
            bins,
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

/// Create `scales/<factor>/mean` and fill it slab by slab. Returns the level
/// length in bins.
fn write_level(store: &PbzStore, t: &Track, factor: u64) -> Result<u64> {
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
    let source_fill = source_fill_as_f64(t)?;
    let nan_aware = t.dtype() == Dtype::F32 && source_fill.is_nan();
    let level_fill = FillValue::from(source_fill as f32);

    let chunk_len = LEVEL_CHUNK_LEN.min(level_len.max(1));
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

    match t.dtype() {
        Dtype::I32 => compute_level::<i32>(t, &level, factor, false, |v| v as f64),
        Dtype::F32 => compute_level::<f32>(t, &level, factor, nan_aware, |v| v as f64),
        Dtype::Bool => {
            compute_level::<bool>(t, &level, factor, false, |v| if v { 1.0 } else { 0.0 })
        }
        other => Err(PbzError::InvalidDtype {
            dtype: format!("scale: unsupported source dtype {other}"),
        }),
    }?;

    Ok(level_len)
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

/// Fill one level array with per-bin means, sequence by sequence, slab by
/// slab. Slabs start on bin boundaries and each bin is contained in exactly
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
