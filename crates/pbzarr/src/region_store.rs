//! Region-store builder: gather many disjoint regions of a source store into a
//! compact region-mode ("peak") store.
//!
//! The output track's `Genome` has one contig per region (length = region
//! length), so the existing import pipeline drives the build unchanged:
//! `Genome::offsets()` is exactly the region-mode `offsets`, task partitioning
//! comes from the output chunk grid, and each per-region fill is one
//! [`RegionReader::read_into`] handed that region's slice. A single
//! `RegionReader` fills the whole column block (contrast `stack`, one reader per
//! column), so a 2D source decodes once per task, not once per column.
//!
//! Rust only *writes* region-mode; Python owns the query/read side.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ndarray::{ArrayD, ArrayViewMut2, Ix1, Ix2, s};

use crate::genome::{Contig, ContigId, Genome, Region};
use crate::import::{Config, Report, run_pipeline};
use crate::io::error::Result as IoResult;
use crate::io::{Dtype, Numeric, ReaderError, ValueReader};
use crate::stack::ProgressFactory;
use crate::store::Segmentation;
use crate::{PbzError, PbzStore, Result, Track, TrackConfig};

/// A decoded source slab held in a [`RegionReader`] cache:
/// `(source contig, window start, window end, data)`.
type CachedSlab<T> = (ContigId, u64, u64, ArrayD<T>);

/// A [`ValueReader`] that fills an output region's rows from a source track,
/// mapping each output region back to its source span and streaming the source
/// a chunk at a time so adjacent regions reuse one decoded slab.
///
/// The driving genome (`out_genome`) has one contig per region; `read_into` is
/// therefore called once per region with the full-width `dst`.
pub struct RegionReader<T> {
    source: Arc<PbzStore>,
    track: String,
    /// One contig per region; drives the pipeline. `read_into`'s `contig_name`
    /// is a region and resolves here to the region index.
    out_genome: Arc<Genome>,
    /// Source genome, for clamping the streaming read window to a contig end.
    src_genome: Arc<Genome>,
    /// Region index -> (source contig, source-contig-local start), region order.
    src_of_region: Arc<Vec<(ContigId, u64)>>,
    /// Number of columns this reader fills (source track column count).
    n_fields: usize,
    /// Source track rank (1 or 2), selecting the copy shape.
    src_rank: usize,
    /// Source track position chunk size, the streaming read granularity.
    src_chunk: u64,
    /// Last decoded source slab. `Mutex` (not `RefCell`) so the type stays
    /// `Sync` for the pipeline; each worker forks its own reader, so the lock is
    /// uncontended.
    cache: Mutex<Option<CachedSlab<T>>>,
    _marker: PhantomData<T>,
}

impl<T: Numeric> ValueReader for RegionReader<T> {
    type Item = T;

    fn contigs(&self) -> &Genome {
        &self.out_genome
    }

    fn n_fields(&self) -> usize {
        self.n_fields
    }

    fn read_into(
        &self,
        region_name: &str,
        local_lo: u64,
        local_hi: u64,
        mut dst: ArrayViewMut2<'_, Self::Item>,
    ) -> IoResult<()> {
        if local_hi <= local_lo {
            return Ok(());
        }
        let ridx = self
            .out_genome
            .id(region_name)
            .ok_or_else(|| other(format!("region {region_name:?} not in region genome")))?
            .as_usize();
        let (src_cid, src_start) = self.src_of_region[ridx];
        let s = src_start + local_lo;
        let e = src_start + local_hi;

        let mut guard = self
            .cache
            .lock()
            .map_err(|_| other("region cache poisoned"))?;
        let hit =
            matches!(guard.as_ref(), Some((c, cs, ce, _)) if *c == src_cid && *cs <= s && e <= *ce);
        if !hit {
            let contig_len = self
                .src_genome
                .get(src_cid)
                .map(|c| c.length)
                .ok_or_else(|| other(format!("source contig {src_cid} vanished")))?;
            let win_start = (s / self.src_chunk) * self.src_chunk;
            let win_end = (e.div_ceil(self.src_chunk) * self.src_chunk).min(contig_len);
            let track = self
                .source
                .track(&self.track)
                .ok_or_else(|| other(format!("source track {:?} vanished", self.track)))?;
            let data = track
                .read_region::<T>(&Region {
                    contig: src_cid,
                    start: win_start,
                    end: win_end,
                })
                .map_err(|e| other(format!("{e}")))?;
            *guard = Some((src_cid, win_start, win_end, data));
        }

        let (cs, data) = match guard.as_ref() {
            Some((_, cs, _, d)) => (*cs, d),
            None => return Err(other("region cache empty after decode")),
        };
        let (a, b) = ((s - cs) as usize, (e - cs) as usize);
        if self.src_rank == 1 {
            let d1 = data
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| other(format!("source rank: {e}")))?;
            dst.slice_mut(s![.., 0]).assign(&d1.slice(s![a..b]));
        } else {
            let d2 = data
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| other(format!("source rank: {e}")))?;
            dst.assign(&d2.slice(s![a..b, ..]));
        }
        Ok(())
    }

    fn fork(&self) -> IoResult<Self> {
        Ok(Self {
            source: Arc::clone(&self.source),
            track: self.track.clone(),
            out_genome: Arc::clone(&self.out_genome),
            src_genome: Arc::clone(&self.src_genome),
            src_of_region: Arc::clone(&self.src_of_region),
            n_fields: self.n_fields,
            src_rank: self.src_rank,
            src_chunk: self.src_chunk,
            cache: Mutex::new(None),
            _marker: PhantomData,
        })
    }
}

fn other(msg: impl Into<String>) -> ReaderError {
    ReaderError::Other(anyhow::anyhow!(msg.into()))
}

/// Configuration for [`build_region_store`].
pub struct RegionBuildConfig {
    /// Tracks to gather. `None` gathers every track of the source store; each
    /// selected track must share the source's reference genome checksum.
    pub tracks: Option<Vec<String>>,
    /// Output position chunk size (default 1M, clamped to the region total).
    pub chunk_size: Option<usize>,
    /// Output position shard size; `None` leaves the output unsharded.
    pub shard_size: Option<usize>,
    /// Output column chunk width for 2D tracks (default: full width).
    pub column_chunk_size: Option<usize>,
    /// Worker threads.
    pub workers: usize,
    /// Optional per-track progress sink factory (see [`crate::stack`]).
    pub progress: Option<Arc<ProgressFactory>>,
}

impl Default for RegionBuildConfig {
    fn default() -> Self {
        Self {
            tracks: None,
            chunk_size: None,
            shard_size: None,
            column_chunk_size: None,
            workers: 4,
            progress: None,
        }
    }
}

/// The region layout derived from an interval query: one entry per region, in
/// sorted flat order, plus the provenance arrays written to disk.
struct RegionLayout {
    genome: Genome,
    src_of_region: Vec<(ContigId, u64)>,
    region_contig: Vec<String>,
    region_start: Vec<i64>,
    region_stop: Vec<i64>,
    region_input_index: Vec<i64>,
}

/// Resolve `intervals` against `ref_genome`, sort by flat start, reject overlaps
/// and empty ranges, and derive the region layout. Mirrors the Python
/// `_reduce.compute_boundaries` + provenance construction.
fn compute_layout(ref_genome: &Genome, intervals: &[(String, u64, u64)]) -> Result<RegionLayout> {
    if intervals.is_empty() {
        return Err(PbzError::InvalidRegion {
            message: "build_region_store: no intervals".into(),
        });
    }
    let offsets = ref_genome.offsets();

    // Resolve each interval to (contig id, contig-local start/end, flat start).
    let mut resolved: Vec<(usize, ContigId, u64, u64, u64)> = Vec::with_capacity(intervals.len());
    for (orig_idx, (name, start, end)) in intervals.iter().enumerate() {
        if end <= start {
            return Err(PbzError::InvalidRegion {
                message: format!("interval {name}:{start}-{end}: start must be < end"),
            });
        }
        let cid = ref_genome
            .id(name)
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: name.clone(),
                available: ref_genome
                    .contigs()
                    .iter()
                    .map(|c| c.name.clone())
                    .collect(),
            })?;
        let flat_start = offsets[cid.as_usize()] as u64 + start;
        resolved.push((orig_idx, cid, *start, *end, flat_start));
    }

    // Stable sort by flat start, preserving input order for ties.
    resolved.sort_by_key(|r| r.4);

    let k = resolved.len();
    let mut genome_contigs = Vec::with_capacity(k);
    let mut src_of_region = Vec::with_capacity(k);
    let mut region_contig = Vec::with_capacity(k);
    let mut region_start = Vec::with_capacity(k);
    let mut region_stop = Vec::with_capacity(k);
    let mut region_input_index = Vec::with_capacity(k);
    let mut prev_flat_end: Option<u64> = None;

    for (i, &(orig_idx, cid, start, end, flat_start)) in resolved.iter().enumerate() {
        let base = offsets[cid.as_usize()] as u64;
        let contig_len = offsets[cid.as_usize() + 1] as u64 - base;
        let clamped_end = end.min(contig_len);
        let flat_end = base + clamped_end;
        if let Some(pe) = prev_flat_end
            && flat_start < pe
        {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "overlapping intervals after sort at position {i}: flat start {flat_start} < previous flat end {pe}"
                ),
            });
        }
        prev_flat_end = Some(flat_end);

        let length = clamped_end - start;
        genome_contigs.push(Contig {
            name: i.to_string(),
            length,
        });
        src_of_region.push((cid, start));
        region_contig.push(
            ref_genome
                .get(cid)
                .map(|c| c.name.clone())
                .unwrap_or_default(),
        );
        region_start.push(start as i64);
        region_stop.push(end as i64);
        region_input_index.push(orig_idx as i64);
    }

    Ok(RegionLayout {
        genome: Genome::new(genome_contigs)?,
        src_of_region,
        region_contig,
        region_start,
        region_stop,
        region_input_index,
    })
}

/// Build a region-mode ("peak") store `out` by gathering `intervals` out of the
/// on-disk `source` store. Selected tracks must share the reference genome
/// (that of the first selected track). Each output track is sized `Σ(region
/// lengths)`, rank-faithful to its source (scalar or 2D), and carries region
/// provenance instead of a `contigs` array.
pub fn build_region_store(
    source: Arc<PbzStore>,
    intervals: &[(String, u64, u64)],
    out: &mut PbzStore,
    config: RegionBuildConfig,
) -> Result<Report> {
    let names: Vec<String> = match &config.tracks {
        Some(ts) => ts.clone(),
        None => source.track_names().map(|s| s.to_owned()).collect(),
    };
    if names.is_empty() {
        return Err(PbzError::Metadata("build_region_store: no tracks".into()));
    }

    let ref_genome = {
        let t0 = source.track(&names[0]).ok_or_else(|| {
            PbzError::Metadata(format!("build_region_store: track {:?} missing", names[0]))
        })?;
        Arc::clone(t0.genome())
    };
    let ref_checksum = ref_genome.checksum();

    let layout = compute_layout(&ref_genome, intervals)?;
    let out_genome = Arc::new(layout.genome.clone());
    let src_of_region = Arc::new(layout.src_of_region.clone());

    out.mark_region_segmentation()?;

    let total_regions = out_genome.len();
    let pos_chunk = config.chunk_size.unwrap_or(1_000_000);

    let mut bytes_written = 0u64;
    let mut tasks_completed = 0usize;

    for name in &names {
        let (dtype, rank, n_cols, src_chunk, labels, col_dim) = {
            let t = source.track(name).ok_or_else(|| {
                PbzError::Metadata(format!("build_region_store: track {name:?} missing"))
            })?;
            if t.genome().checksum() != ref_checksum {
                return Err(PbzError::Metadata(format!(
                    "build_region_store: track {name:?} genome differs from {:?}",
                    names[0]
                )));
            }
            let rank = t.rank();
            let labels = if rank == 2 {
                t.column_labels()?
            } else {
                Vec::new()
            };
            let col_dim = t.column_dim().map(|s| s.to_owned());
            (
                t.dtype(),
                rank,
                t.columns_count()?,
                t.chunk_size()? as u64,
                labels,
                col_dim,
            )
        };

        let mut track_cfg = TrackConfig::new(dtype).chunk_size(pos_chunk);
        if rank == 2 {
            let col_dim = col_dim.unwrap_or_else(|| "column".to_owned());
            let col_chunk = config.column_chunk_size.unwrap_or(n_cols);
            track_cfg = track_cfg
                .columns(labels)
                .column_dim(col_dim)
                .column_chunk_size(col_chunk);
            if let Some(sh) = config.shard_size {
                track_cfg = track_cfg.shard_size(sh).shard_column_size(n_cols);
            }
        } else if let Some(sh) = config.shard_size {
            track_cfg = track_cfg.shard_size(sh);
        }

        let seg = Segmentation::Region {
            region_contig: layout.region_contig.clone(),
            region_start: layout.region_start.clone(),
            region_stop: layout.region_stop.clone(),
            region_input_index: layout.region_input_index.clone(),
            parent_checksum: ref_checksum.clone(),
        };

        let sink = config.progress.as_ref().map(|make| {
            let total = out_genome.contigs().iter().map(|c| c.length).sum::<u64>()
                * n_cols as u64
                * dtype.size_bytes() as u64;
            make(name, total)
        });
        let pcfg = Config {
            workers: config.workers,
            progress: sink,
            ..Config::default()
        };

        let out_track = out.create_region_track(name, layout.genome.clone(), track_cfg, seg)?;
        let report = dispatch(
            dtype,
            &source,
            name,
            &out_genome,
            &src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            &pcfg,
        )?;
        bytes_written += report.bytes_written;
        tasks_completed += report.tasks_completed;
    }

    Ok(Report {
        contigs_written: total_regions,
        bytes_written,
        tasks_completed,
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    dtype: Dtype,
    source: &Arc<PbzStore>,
    track: &str,
    out_genome: &Arc<Genome>,
    src_of_region: &Arc<Vec<(ContigId, u64)>>,
    n_cols: usize,
    rank: usize,
    src_chunk: u64,
    out_track: &Track,
    cfg: &Config,
) -> Result<Report> {
    match dtype {
        Dtype::U8 => build_one::<u8>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::U16 => build_one::<u16>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::U32 => build_one::<u32>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::I8 => build_one::<i8>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::I16 => build_one::<i16>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::I32 => build_one::<i32>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::F32 => build_one::<f32>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::F64 => build_one::<f64>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
        Dtype::Bool => build_one::<bool>(
            source,
            track,
            out_genome,
            src_of_region,
            n_cols,
            rank,
            src_chunk,
            out_track,
            cfg,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_one<T: Numeric>(
    source: &Arc<PbzStore>,
    track: &str,
    out_genome: &Arc<Genome>,
    src_of_region: &Arc<Vec<(ContigId, u64)>>,
    n_cols: usize,
    rank: usize,
    src_chunk: u64,
    out_track: &Track,
    cfg: &Config,
) -> Result<Report> {
    let src_genome = {
        let t = source
            .track(track)
            .ok_or_else(|| PbzError::Metadata(format!("track {track:?} missing")))?;
        Arc::clone(t.genome())
    };
    let reader = RegionReader::<T> {
        source: Arc::clone(source),
        track: track.to_owned(),
        out_genome: Arc::clone(out_genome),
        src_genome,
        src_of_region: Arc::clone(src_of_region),
        n_fields: n_cols,
        src_rank: rank,
        src_chunk: src_chunk.max(1),
        cache: Mutex::new(None),
        _marker: PhantomData,
    };
    run_pipeline::<T, _>(out_track, vec![reader], cfg)
}
