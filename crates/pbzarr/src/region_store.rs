//! Region-store builder: gather many disjoint regions of a source store into a
//! compact region-mode ("peak") store.
//!
//! The output track's `Genome` has one contig per region (length = region
//! length). The builder plans work by touched source chunks, decodes each source
//! chunk once per output track, assembles complete output write units in order,
//! then compresses/writes those units in parallel through `Track::write_flat`.
//!
//! Rust only *writes* region-mode; Python owns the query/read side.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::bounded;
use ndarray::{Array2, ArrayD, Axis, Ix1, Ix2, s};

use crate::genome::{Contig, ContigId, Genome, Region};
use crate::import::{ProgressSink, Report};
use crate::io::{Dtype, Numeric};
use crate::stack::ProgressFactory;
use crate::store::Segmentation;
use crate::{PbzError, PbzStore, Result, Track, TrackConfig};

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
    /// Source decode worker threads. `None` derives a split from `workers`.
    pub decode_workers: Option<usize>,
    /// Output write/compression worker threads. `None` derives a split from `workers`.
    pub write_workers: Option<usize>,
    /// Maximum complete output write-unit buffers queued for writer workers.
    pub writer_queue_depth: usize,
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
            decode_workers: None,
            write_workers: None,
            writer_queue_depth: 2,
            progress: None,
        }
    }
}

fn worker_split(
    config: &RegionBuildConfig,
    source_task_count: usize,
    write_task_count: usize,
) -> (usize, usize) {
    let total = config.workers.max(1);
    let source_cap = source_task_count.max(1);
    let write_cap = write_task_count.max(1);

    match (config.decode_workers, config.write_workers) {
        (Some(d), Some(w)) => (d.max(1).min(source_cap), w.max(1).min(write_cap)),
        (Some(d), None) => {
            let d = d.max(1).min(source_cap);
            let w = total.saturating_sub(d).max(1).min(write_cap);
            (d, w)
        }
        (None, Some(w)) => {
            let w = w.max(1).min(write_cap);
            let d = total.saturating_sub(w).max(1).min(source_cap);
            (d, w)
        }
        (None, None) => {
            let w = (total / 2).max(1).min(write_cap);
            let d = total.saturating_sub(w).max(1).min(source_cap);
            (d, w)
        }
    }
}

struct WriteUnitBuffer<T: Numeric> {
    start: u64,
    end: u64,
    rows: Array2<T>,
}

impl<T: Numeric> WriteUnitBuffer<T> {
    fn new(start: u64, end: u64, n_cols: usize) -> Self {
        let len = (end - start) as usize;
        Self {
            start,
            end,
            rows: Array2::from_elem((len, n_cols), T::ZERO),
        }
    }

    fn into_array(self, rank: usize) -> Result<ArrayD<T>> {
        match rank {
            1 => Ok(self.rows.remove_axis(Axis(1)).into_dyn()),
            2 => Ok(self.rows.into_dyn()),
            _ => Err(PbzError::Metadata(format!(
                "region gather: unsupported rank {rank}"
            ))),
        }
    }
}

fn copy_piece<T: Numeric>(
    decoded: &ArrayD<T>,
    source_chunk_start: u64,
    piece: &CopyPiece,
    rank: usize,
    dst: &mut WriteUnitBuffer<T>,
) -> Result<()> {
    let source_lo = (piece.source_start - source_chunk_start) as usize;
    let source_hi = (piece.source_end - source_chunk_start) as usize;
    let dst_lo = (piece.output_start - dst.start) as usize;
    let dst_hi = (piece.output_end - dst.start) as usize;

    match rank {
        1 => {
            let src = decoded
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|e| PbzError::Metadata(format!("region gather source rank: {e}")))?;
            dst.rows
                .slice_mut(s![dst_lo..dst_hi, 0])
                .assign(&src.slice(s![source_lo..source_hi]));
        }
        2 => {
            let src = decoded
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| PbzError::Metadata(format!("region gather source rank: {e}")))?;
            dst.rows
                .slice_mut(s![dst_lo..dst_hi, ..])
                .assign(&src.slice(s![source_lo..source_hi, ..]));
        }
        _ => {
            return Err(PbzError::Metadata(format!(
                "region gather: unsupported rank {rank}"
            )));
        }
    }
    Ok(())
}

struct GatherConfig {
    decode_workers: usize,
    write_workers: usize,
    writer_queue_depth: usize,
    progress: Option<Arc<dyn ProgressSink>>,
}

struct DecodedTask<T: Numeric> {
    index: usize,
    task: SourceChunkTask,
    data: ArrayD<T>,
}

struct WriteTask<T: Numeric> {
    start: u64,
    end: u64,
    data: ArrayD<T>,
    bytes: u64,
}

struct GatherState {
    bytes_written: AtomicU64,
    tasks_completed: AtomicUsize,
    first_err: Mutex<Option<PbzError>>,
}

impl GatherState {
    fn new() -> Self {
        Self {
            bytes_written: AtomicU64::new(0),
            tasks_completed: AtomicUsize::new(0),
            first_err: Mutex::new(None),
        }
    }

    fn record_err(&self, err: PbzError) {
        if let Ok(mut slot) = self.first_err.lock()
            && slot.is_none()
        {
            *slot = Some(err);
        }
    }

    fn has_err(&self) -> bool {
        self.first_err
            .lock()
            .map(|slot| slot.is_some())
            .unwrap_or(true)
    }
}

fn decode_source_task<T: Numeric>(
    source: &PbzStore,
    track_name: &str,
    task: SourceChunkTask,
) -> Result<DecodedTask<T>> {
    let track = source
        .track(track_name)
        .ok_or_else(|| PbzError::Metadata(format!("source track {track_name:?} missing")))?;
    let contig_len = track
        .genome()
        .get(task.source_contig)
        .map(|c| c.length)
        .ok_or_else(|| {
            PbzError::Metadata(format!("source contig {} missing", task.source_contig))
        })?;
    let end = task.chunk_end.min(contig_len);
    let data = track.read_region::<T>(&Region {
        contig: task.source_contig,
        start: task.chunk_start,
        end,
    })?;
    Ok(DecodedTask {
        index: 0,
        task,
        data,
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_decoded_task<T: Numeric>(
    decoded: DecodedTask<T>,
    pending_buffer: &mut Option<WriteUnitBuffer<T>>,
    write_unit: u64,
    total_len: u64,
    rank: usize,
    n_cols: usize,
    dtype_size: usize,
    write_tx: &crossbeam_channel::Sender<WriteTask<T>>,
) -> Result<()> {
    for piece in &decoded.task.pieces {
        let unit_start = (piece.write_unit_index as u64) * write_unit;
        let unit_end = (unit_start + write_unit).min(total_len);

        let needs_new_buffer = pending_buffer
            .as_ref()
            .map(|buf| buf.start != unit_start)
            .unwrap_or(true);
        if needs_new_buffer {
            if let Some(buf) = pending_buffer.take() {
                let start = buf.start;
                let end = buf.end;
                let bytes = (end - start) * n_cols as u64 * dtype_size as u64;
                write_tx
                    .send(WriteTask {
                        start,
                        end,
                        data: buf.into_array(rank)?,
                        bytes,
                    })
                    .map_err(|e| PbzError::Store(format!("region gather send write task: {e}")))?;
            }
            *pending_buffer = Some(WriteUnitBuffer::new(unit_start, unit_end, n_cols));
        }

        let buf = pending_buffer
            .as_mut()
            .ok_or_else(|| PbzError::Metadata("region gather buffer missing".into()))?;
        copy_piece(&decoded.data, decoded.task.chunk_start, piece, rank, buf)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_region_gather_pipeline<T: Numeric>(
    source: &Arc<PbzStore>,
    track_name: &str,
    out_track: &Track,
    layout: &RegionLayout,
    source_chunk: u64,
    rank: usize,
    n_cols: usize,
    dtype_size: usize,
    config: GatherConfig,
) -> Result<Report> {
    let write_unit = out_track.chunk_size()? as u64;
    let total_len = out_track.total_len();
    let source_tasks = layout.source_chunk_tasks(source_chunk.max(1), write_unit.max(1));

    let (source_tx, source_rx) =
        bounded::<(usize, SourceChunkTask)>((config.decode_workers * 2).max(1));
    let (decoded_tx, decoded_rx) = bounded::<DecodedTask<T>>((config.decode_workers * 2).max(1));
    let (write_tx, write_rx) = bounded::<WriteTask<T>>(config.writer_queue_depth.max(1));
    let state = Arc::new(GatherState::new());

    thread::scope(|scope| {
        for _ in 0..config.decode_workers.max(1) {
            let source_rx = source_rx.clone();
            let decoded_tx = decoded_tx.clone();
            let source = Arc::clone(source);
            let track_name = track_name.to_owned();
            let state = Arc::clone(&state);
            scope.spawn(move || {
                while let Ok((index, task)) = source_rx.recv() {
                    if state.has_err() {
                        continue;
                    }
                    match decode_source_task::<T>(&source, &track_name, task) {
                        Ok(mut decoded) => {
                            decoded.index = index;
                            if let Err(e) = decoded_tx.send(decoded) {
                                state.record_err(PbzError::Store(format!(
                                    "region gather send decoded task: {e}"
                                )));
                                break;
                            }
                        }
                        Err(e) => state.record_err(e),
                    }
                }
            });
        }
        drop(decoded_tx);

        for _ in 0..config.write_workers.max(1) {
            let write_rx = write_rx.clone();
            let state = Arc::clone(&state);
            let progress = config.progress.clone();
            scope.spawn(move || {
                while let Ok(task) = write_rx.recv() {
                    if state.has_err() {
                        continue;
                    }
                    match out_track.write_flat::<T>(task.start, task.end, task.data) {
                        Ok(()) => {
                            state.bytes_written.fetch_add(task.bytes, Ordering::Relaxed);
                            state.tasks_completed.fetch_add(1, Ordering::Relaxed);
                            if let Some(p) = progress.as_deref() {
                                p.tick(task.bytes);
                            }
                        }
                        Err(e) => state.record_err(e),
                    }
                }
            });
        }

        let assembler_state = Arc::clone(&state);
        scope.spawn(move || {
            let mut next_index = 0usize;
            let mut waiting = BTreeMap::<usize, DecodedTask<T>>::new();
            let mut pending_buffer = None;

            while let Ok(decoded) = decoded_rx.recv() {
                waiting.insert(decoded.index, decoded);
                while let Some(decoded) = waiting.remove(&next_index) {
                    if assembler_state.has_err() {
                        next_index += 1;
                        continue;
                    }
                    if let Err(e) = assemble_decoded_task(
                        decoded,
                        &mut pending_buffer,
                        write_unit,
                        total_len,
                        rank,
                        n_cols,
                        dtype_size,
                        &write_tx,
                    ) {
                        assembler_state.record_err(e);
                        break;
                    }
                    next_index += 1;
                }
            }

            if !assembler_state.has_err()
                && let Some(buf) = pending_buffer.take()
            {
                let start = buf.start;
                let end = buf.end;
                let bytes = (end - start) * n_cols as u64 * dtype_size as u64;
                match buf.into_array(rank) {
                    Ok(data) => {
                        if let Err(e) = write_tx.send(WriteTask {
                            start,
                            end,
                            data,
                            bytes,
                        }) {
                            assembler_state.record_err(PbzError::Store(format!(
                                "region gather send final write task: {e}"
                            )));
                        }
                    }
                    Err(e) => assembler_state.record_err(e),
                }
            }
            drop(write_tx);
        });

        for (index, task) in source_tasks.into_iter().enumerate() {
            if state.has_err() {
                break;
            }
            if source_tx.send((index, task)).is_err() {
                break;
            }
        }
        drop(source_tx);
    });

    if let Some(ref p) = config.progress {
        p.done();
    }
    if let Some(e) = state
        .first_err
        .lock()
        .map_err(|_| PbzError::Metadata("region gather error slot poisoned".into()))?
        .take()
    {
        return Err(e);
    }

    Ok(Report {
        contigs_written: layout.genome.iter().filter(|(_, c)| c.length > 0).count(),
        bytes_written: state.bytes_written.load(Ordering::Relaxed),
        tasks_completed: state.tasks_completed.load(Ordering::Relaxed),
    })
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
    offsets: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyPiece {
    source_start: u64,
    source_end: u64,
    output_start: u64,
    output_end: u64,
    write_unit_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceChunkTask {
    source_contig: ContigId,
    chunk_start: u64,
    chunk_end: u64,
    pieces: Vec<CopyPiece>,
}

impl RegionLayout {
    fn source_chunk_tasks(&self, src_chunk: u64, write_unit: u64) -> Vec<SourceChunkTask> {
        let src_chunk = src_chunk.max(1);
        let write_unit = write_unit.max(1);
        let mut tasks: Vec<SourceChunkTask> = Vec::new();

        for (region_index, &(source_contig, region_source_start)) in
            self.src_of_region.iter().enumerate()
        {
            let mut source_cursor = region_source_start;
            let mut output_cursor = self.offsets[region_index];
            let output_end = self.offsets[region_index + 1];

            while output_cursor < output_end {
                let chunk_start = (source_cursor / src_chunk) * src_chunk;
                let chunk_end = chunk_start + src_chunk;
                let write_unit_index = (output_cursor / write_unit) as usize;
                let write_unit_end = ((write_unit_index as u64) + 1) * write_unit;
                let step = (chunk_end - source_cursor)
                    .min(write_unit_end - output_cursor)
                    .min(output_end - output_cursor);

                let piece = CopyPiece {
                    source_start: source_cursor,
                    source_end: source_cursor + step,
                    output_start: output_cursor,
                    output_end: output_cursor + step,
                    write_unit_index,
                };

                match tasks.last_mut() {
                    Some(last)
                        if last.source_contig == source_contig
                            && last.chunk_start == chunk_start =>
                    {
                        last.pieces.push(piece);
                    }
                    _ => tasks.push(SourceChunkTask {
                        source_contig,
                        chunk_start,
                        chunk_end,
                        pieces: vec![piece],
                    }),
                }

                source_cursor += step;
                output_cursor += step;
            }
        }

        tasks
    }
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
    let mut out_offsets = Vec::with_capacity(k + 1);
    out_offsets.push(0);
    let mut output_cursor = 0u64;
    let mut prev_flat_end: Option<u64> = None;

    for (i, &(orig_idx, cid, start, end, flat_start)) in resolved.iter().enumerate() {
        let base = offsets[cid.as_usize()] as u64;
        let contig_len = offsets[cid.as_usize() + 1] as u64 - base;
        let clamped_end = end.min(contig_len);
        if start >= clamped_end {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "interval {}:{start}-{end} is empty after clamping to contig length {contig_len}",
                    ref_genome
                        .get(cid)
                        .map(|c| c.name.as_str())
                        .unwrap_or("<unknown>")
                ),
            });
        }
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
        output_cursor += length;
        out_offsets.push(output_cursor);
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
        offsets: out_offsets,
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

        let out_track = out.create_region_track(name, layout.genome.clone(), track_cfg, seg)?;
        let write_unit = out_track.chunk_size()? as u64;
        let source_task_count = layout
            .source_chunk_tasks(src_chunk.max(1), write_unit.max(1))
            .len();
        let write_task_count = out_track.total_len().div_ceil(write_unit.max(1)) as usize;
        let (decode_workers, write_workers) =
            worker_split(&config, source_task_count, write_task_count);
        let gather_cfg = GatherConfig {
            decode_workers,
            write_workers,
            writer_queue_depth: config.writer_queue_depth.max(1),
            progress: sink,
        };
        let report = dispatch(
            dtype, &source, name, &layout, n_cols, rank, src_chunk, out_track, gather_cfg,
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
    layout: &RegionLayout,
    n_cols: usize,
    rank: usize,
    src_chunk: u64,
    out_track: &Track,
    cfg: GatherConfig,
) -> Result<Report> {
    match dtype {
        Dtype::U8 => build_one::<u8>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::U16 => build_one::<u16>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::U32 => build_one::<u32>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::I8 => build_one::<i8>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::I16 => build_one::<i16>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::I32 => build_one::<i32>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::F32 => build_one::<f32>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::F64 => build_one::<f64>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
        Dtype::Bool => build_one::<bool>(
            source, track, layout, n_cols, rank, src_chunk, out_track, cfg,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_one<T: Numeric>(
    source: &Arc<PbzStore>,
    track: &str,
    layout: &RegionLayout,
    n_cols: usize,
    rank: usize,
    src_chunk: u64,
    out_track: &Track,
    cfg: GatherConfig,
) -> Result<Report> {
    run_region_gather_pipeline::<T>(
        source,
        track,
        out_track,
        layout,
        src_chunk.max(1),
        rank,
        n_cols,
        T::DTYPE.size_bytes(),
        cfg,
    )
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn test_genome() -> Genome {
        Genome::new(vec![
            Contig {
                name: "chr1".to_owned(),
                length: 40,
            },
            Contig {
                name: "chr2".to_owned(),
                length: 30,
            },
        ])
        .unwrap()
    }

    #[test]
    fn layout_records_output_offsets_in_sorted_region_order() {
        let intervals = vec![
            ("chr1".to_owned(), 15, 25),
            ("chr2".to_owned(), 5, 14),
            ("chr1".to_owned(), 3, 8),
        ];

        let layout = compute_layout(&test_genome(), &intervals).unwrap();

        assert_eq!(layout.offsets, vec![0, 5, 15, 24]);
        assert_eq!(layout.region_contig, vec!["chr1", "chr1", "chr2"]);
        assert_eq!(layout.region_start, vec![3, 15, 5]);
        assert_eq!(layout.region_stop, vec![8, 25, 14]);
        assert_eq!(layout.region_input_index, vec![2, 0, 1]);
    }

    #[test]
    fn layout_rejects_interval_empty_after_contig_clamp() {
        let err = match compute_layout(&test_genome(), &[("chr1".to_owned(), 40, 45)]) {
            Err(err) => err,
            Ok(_) => panic!("empty clamped interval should be rejected"),
        };

        assert!(format!("{err}").contains("empty after clamping"));
    }

    #[test]
    fn layout_splits_copy_pieces_on_source_chunks_and_output_write_units() {
        let intervals = vec![
            ("chr1".to_owned(), 15, 25),
            ("chr2".to_owned(), 5, 14),
            ("chr1".to_owned(), 3, 8),
        ];
        let layout = compute_layout(&test_genome(), &intervals).unwrap();

        let tasks = layout.source_chunk_tasks(10, 8);

        let keys: Vec<(usize, u64, u64)> = tasks
            .iter()
            .map(|t| (t.source_contig.as_usize(), t.chunk_start, t.chunk_end))
            .collect();
        assert_eq!(
            keys,
            vec![
                (0, 0, 10),
                (0, 10, 20),
                (0, 20, 30),
                (1, 0, 10),
                (1, 10, 20)
            ]
        );

        let pieces: Vec<(u64, u64, u64, u64, usize)> = tasks
            .iter()
            .flat_map(|t| {
                t.pieces.iter().map(|p| {
                    (
                        p.source_start,
                        p.source_end,
                        p.output_start,
                        p.output_end,
                        p.write_unit_index,
                    )
                })
            })
            .collect();
        assert_eq!(
            pieces,
            vec![
                (3, 8, 0, 5, 0),
                (15, 18, 5, 8, 0),
                (18, 20, 8, 10, 1),
                (20, 25, 10, 15, 1),
                (5, 6, 15, 16, 1),
                (6, 10, 16, 20, 2),
                (10, 14, 20, 24, 2),
            ]
        );
    }
}

#[cfg(test)]
mod gather_unit_tests {
    use super::*;
    use ndarray::{Array1, Array2, Ix1, Ix2};

    #[test]
    fn worker_split_clamps_to_task_counts() {
        let cfg = RegionBuildConfig {
            workers: 512,
            ..Default::default()
        };
        assert_eq!(worker_split(&cfg, 4, 3), (4, 3));
    }

    #[test]
    fn worker_split_honors_explicit_writer_count() {
        let cfg = RegionBuildConfig {
            workers: 12,
            write_workers: Some(5),
            ..Default::default()
        };
        assert_eq!(worker_split(&cfg, 20, 20), (7, 5));
    }

    #[test]
    fn copy_piece_copies_scalar_rows_into_output_buffer() {
        let decoded = Array1::from_iter(100..110i32).into_dyn();
        let piece = CopyPiece {
            source_start: 13,
            source_end: 17,
            output_start: 5,
            output_end: 9,
            write_unit_index: 0,
        };
        let mut out = WriteUnitBuffer::<i32>::new(0, 10, 1);

        copy_piece(&decoded, 10, &piece, 1, &mut out).unwrap();

        let got = out
            .into_array(1)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert_eq!(
            got.slice(ndarray::s![5..9]).to_vec(),
            vec![103, 104, 105, 106]
        );
    }

    #[test]
    fn copy_piece_copies_2d_rows_and_all_columns() {
        let decoded = Array2::from_shape_fn((6, 3), |(r, c)| (r as i32 * 10) + c as i32);
        let piece = CopyPiece {
            source_start: 22,
            source_end: 25,
            output_start: 8,
            output_end: 11,
            write_unit_index: 1,
        };
        let mut out = WriteUnitBuffer::<i32>::new(8, 16, 3);

        copy_piece(&decoded.into_dyn(), 20, &piece, 2, &mut out).unwrap();

        let got = out
            .into_array(2)
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(
            got.slice(ndarray::s![0..3, ..]).to_owned(),
            ndarray::arr2(&[[20, 21, 22], [30, 31, 32], [40, 41, 42]])
        );
    }
}
