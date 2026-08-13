//! Generic import pipeline. Drives `ValueReader` sources into a `Track` via a
//! scoped worker pool; each worker forks its readers, then reads + writes one
//! task directly.
//!
//! Each task is one physical write unit: a chunk-aligned segment of the flat
//! position axis at full column width (the shard, when the track is sharded).
//! Task boundaries land on the chunk grid, so every task writes a whole chunk
//! that no other task touches. That keeps zarrs on its single-encode path and
//! removes the concurrent read-modify-write hazard entirely, so no write lock
//! is needed. A task may straddle contig boundaries; the reader fill loop
//! visits each overlapping contig by name.

use std::mem;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::bounded;
use ndarray::{Array2, Axis};

use crate::Result;
use crate::error::PbzError;
use crate::io::{
    ColumnBuffer, ColumnSinkMut, Dtype, MatrixBuffer, MultiValueReader, Numeric, ValueReader,
};
use crate::track::Track;

/// Hook for callers to observe pipeline progress. Implementations must be
/// `Send + Sync` because workers call `tick` concurrently.
pub trait ProgressSink: Send + Sync {
    fn tick(&self, _bytes: u64) {}
    fn done(&self) {}
}

/// Configuration for `run_pipeline`.
pub struct Config {
    /// Number of reader/writer worker threads.
    pub workers: usize,
    /// Position chunk size for the track being imported. Consumed by the
    /// format readers (`from_d4`/`from_bigwig`) when they create the track;
    /// `run_pipeline` always steps by the track's on-disk write unit.
    pub chunk_size: Option<usize>,
    /// Column chunk size for the track being imported. Consumed at track
    /// creation, like `chunk_size`.
    pub column_chunk_size: Option<usize>,
    /// Position shard size for the track being imported. Consumed at track
    /// creation, like `chunk_size`; `None` leaves the track unsharded.
    pub shard_size: Option<usize>,
    /// Column shard size for the track being imported. Consumed at track
    /// creation; ignored unless `shard_size` is set.
    pub shard_column_size: Option<usize>,
    /// Column-axis dimension name for a cohort import (several sources). The
    /// axis is generic; the readers default it to `"sample"`, but set this to
    /// `"strand"`, `"context"`, etc. when the columns are not samples. Ignored
    /// for single-source (scalar) imports.
    pub column_dim: Option<String>,
    /// Optional progress observer.
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 4,
            chunk_size: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            column_dim: None,
            progress: None,
        }
    }
}

impl Config {
    /// Build a `TrackConfig` for `dtype`/`fill` (`None` leaves the track's
    /// default fill unset), applying this config's chunk/shard sizing.
    /// `labels` being `Some` is what makes the track 2D (callers decide via
    /// `n_sources > 1 || self.column_dim.is_some()`, so a single source with
    /// an explicit `column_dim` still gets a 1-column track);
    /// `default_column_dim` is the fallback when `column_dim` is unset.
    /// Shared by every format reader's import entry point.
    pub fn track_config(
        &self,
        dtype: crate::io::Dtype,
        fill: Option<serde_json::Value>,
        labels: Option<Vec<String>>,
        default_column_dim: &str,
    ) -> crate::track::TrackConfig {
        let mut cfg = crate::track::TrackConfig::new(dtype);
        if let Some(fill) = fill {
            cfg = cfg.fill_value(fill);
        }
        if let Some(cs) = self.chunk_size {
            cfg = cfg.chunk_size(cs);
        }
        if let Some(ss) = self.shard_size {
            cfg = cfg.shard_size(ss);
        }
        if let Some(scs) = self.shard_column_size {
            cfg = cfg.shard_column_size(scs);
        }
        if let Some(labels) = labels {
            let dim = self.column_dim.as_deref().unwrap_or(default_column_dim);
            cfg = cfg.columns(labels).column_dim(dim);
            if let Some(ccs) = self.column_chunk_size {
                cfg = cfg.column_chunk_size(ccs);
            }
        }
        cfg
    }
}

/// Summary returned by `run_pipeline` on success.
pub struct Report {
    pub contigs_written: usize,
    pub bytes_written: u64,
    /// Number of chunk tasks that completed successfully.
    pub tasks_completed: usize,
    /// Number of chunk tasks skipped because no reader covered them
    /// (`may_have_data` was false for every overlapping contig).
    pub tasks_skipped: usize,
}

/// One physical position-by-column write unit. Boundaries land on the outer
/// chunk grid, which is the shard grid when sharding is enabled.
#[derive(Clone, Debug)]
struct ChunkTask {
    start: u64,
    end: u64,
    columns: std::ops::Range<u64>,
}

/// One contig's overlap with a task, precomputed so the `may_have_data` skip
/// check can run before any buffer is allocated.
struct ContigWindow<'g> {
    name: &'g str,
    row_lo: usize,
    row_hi: usize,
    local_lo: u64,
    local_hi: u64,
}

/// True when no window in `windows` has a reader covering it, i.e. this
/// task's whole region is a fill-value gap and both the read and the write
/// can be skipped (safe only because tracks are always created with a zero
/// fill, so an absent chunk already reads back as a zero-filled one would).
/// Bumps `state.tasks_skipped` as a side effect, since every call site does
/// that and returns immediately when this is true.
fn skip_uncovered_task(
    windows: &[ContigWindow],
    state: &State,
    mut may_have_data: impl FnMut(&ContigWindow) -> bool,
) -> bool {
    let covered = windows.iter().any(&mut may_have_data);
    if !covered {
        state.tasks_skipped.fetch_add(1, Ordering::Relaxed);
    }
    !covered
}

fn contig_windows(genome: &crate::genome::Genome, start: u64, end: u64) -> Vec<ContigWindow<'_>> {
    let offsets = genome.offsets();
    let mut windows = Vec::new();
    for (i, contig) in genome.contigs().iter().enumerate() {
        let c_start = offsets[i] as u64;
        if c_start >= end {
            break;
        }
        let c_end = offsets[i + 1] as u64;
        if c_end <= start {
            continue;
        }
        let ov_start = start.max(c_start);
        let ov_end = end.min(c_end);
        windows.push(ContigWindow {
            name: contig.name.as_str(),
            row_lo: (ov_start - start) as usize,
            row_hi: (ov_end - start) as usize,
            local_lo: ov_start - c_start,
            local_hi: ov_end - c_start,
        });
    }
    windows
}

struct State {
    bytes_written: AtomicU64,
    tasks_completed: AtomicUsize,
    tasks_skipped: AtomicUsize,
    first_err: Mutex<Option<PbzError>>,
}

impl State {
    fn record_err(&self, err: PbzError) {
        let mut slot = self.first_err.lock().expect("error slot poisoned");
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    fn has_err(&self) -> bool {
        self.first_err
            .lock()
            .expect("error slot poisoned")
            .is_some()
    }
}

/// Drive a set of `ValueReader` instances into a `Track`, chunk by chunk.
///
/// - For scalar (rank-1) tracks, `readers.len()` MUST be 1.
/// - For cohort (rank-2) tracks, `readers.len()` MUST equal the track's column
///   count (as declared in the on-disk store).
///
/// Workers fork each reader once via `ValueReader::fork`; the original
/// `readers` vec is consumed by the function.
pub fn run_pipeline<T, R>(track: &Track, readers: Vec<R>, config: &Config) -> Result<Report>
where
    T: Numeric,
    R: ValueReader<Item = T>,
{
    if T::DTYPE != track.dtype() {
        return Err(PbzError::InvalidDtype {
            dtype: format!(
                "track {:?} is {} but pipeline got {}",
                track.name(),
                track.dtype(),
                T::DTYPE
            ),
        });
    }

    let n_readers = readers.len();

    if track.rank() == 2 {
        let expected = track.columns_count()?;
        if n_readers != expected {
            return Err(PbzError::Metadata(format!(
                "cohort track {:?} expects {expected} readers; got {n_readers}",
                track.name()
            )));
        }
    } else if n_readers != 1 {
        return Err(PbzError::Metadata(format!(
            "scalar track {:?} expects 1 reader; got {n_readers}",
            track.name()
        )));
    }

    let write_unit = track.write_unit_shape()?;
    let position_step = u64::try_from(write_unit[0])
        .map_err(|_| PbzError::Metadata("position write unit exceeds u64".into()))?;
    let column_step = if track.rank() == 2 {
        u64::try_from(write_unit[1])
            .map_err(|_| PbzError::Metadata("column write unit exceeds u64".into()))?
    } else {
        1
    };
    let genome = Arc::clone(track.genome());
    let total = track.total_len();
    let n_contigs = genome.iter().filter(|(_, c)| c.length > 0).count();

    let mut tasks: Vec<ChunkTask> = Vec::new();
    let n_position_units = total.div_ceil(position_step);
    let n_columns = u64::try_from(track.columns_count()?)
        .map_err(|_| PbzError::Metadata("column count exceeds u64".into()))?;
    for position_index in 0..n_position_units {
        let start = position_index * position_step;
        let end = (start + position_step).min(total);
        for column_start in (0..n_columns).step_by(column_step as usize) {
            let column_end = (column_start + column_step).min(n_columns);
            tasks.push(ChunkTask {
                start,
                end,
                columns: column_start..column_end,
            });
        }
    }

    let workers = config.workers.max(1);
    let task_cap = (workers * 2).max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>(task_cap);

    let readers = Arc::new(readers);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
        first_err: Mutex::new(None),
    });

    thread::scope(|scope| {
        for _ in 0..workers {
            let task_rx = task_rx.clone();
            let readers = Arc::clone(&readers);
            let genome = Arc::clone(&genome);
            let state = Arc::clone(&state);
            let progress = config.progress.clone();
            scope.spawn(move || {
                let mut cached_columns: Option<(std::ops::Range<usize>, Vec<R>)> = None;

                while let Ok(task) = task_rx.recv() {
                    if state.has_err() {
                        // Drain remaining tasks to let the channel close cleanly.
                        continue;
                    }
                    let columns = match usize::try_from(task.columns.start)
                        .ok()
                        .zip(usize::try_from(task.columns.end).ok())
                    {
                        Some((start, end)) => start..end,
                        None => {
                            state.record_err(PbzError::Metadata(
                                "task column range exceeds usize".into(),
                            ));
                            continue;
                        }
                    };
                    let needs_new_forks = cached_columns
                        .as_ref()
                        .is_none_or(|(cached, _)| *cached != columns);
                    if needs_new_forks {
                        let forked = match readers[columns.clone()]
                            .iter()
                            .map(ValueReader::fork)
                            .collect::<crate::io::error::Result<Vec<_>>>()
                        {
                            Ok(forked) => forked,
                            Err(error) => {
                                state.record_err(PbzError::Metadata(format!(
                                    "reader fork failed: {error}"
                                )));
                                continue;
                            }
                        };
                        cached_columns = Some((columns.clone(), forked));
                    }
                    let Some((_, forked)) = cached_columns.as_mut() else {
                        continue;
                    };
                    if let Err(e) = process_task::<T, R>(
                        track,
                        forked,
                        &genome,
                        &task,
                        progress.as_deref(),
                        &state,
                    ) {
                        state.record_err(e);
                    }
                }
            });
        }

        // Push tasks from the main thread; drop the sender to signal workers.
        for task in tasks {
            if state.has_err() {
                break;
            }
            if task_tx.send(task).is_err() {
                break;
            }
        }
        drop(task_tx);
    });

    if let Some(ref p) = config.progress {
        p.done();
    }

    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    Ok(Report {
        contigs_written: n_contigs,
        bytes_written: state.bytes_written.load(Ordering::Relaxed),
        tasks_completed: state.tasks_completed.load(Ordering::Relaxed),
        tasks_skipped: state.tasks_skipped.load(Ordering::Relaxed),
    })
}

fn process_task<T, R>(
    track: &Track,
    forked: &mut [R],
    genome: &crate::genome::Genome,
    task: &ChunkTask,
    progress: Option<&dyn ProgressSink>,
    state: &State,
) -> Result<()>
where
    T: Numeric,
    R: ValueReader<Item = T>,
{
    let (gs, ge) = (task.start, task.end);
    let chunk_len = (ge - gs) as usize;
    let tile_width = usize::try_from(task.columns.end - task.columns.start)
        .map_err(|_| PbzError::Metadata("tile width exceeds usize".into()))?;

    let windows = contig_windows(genome, gs, ge);
    if skip_uncovered_task(&windows, state, |w| {
        forked
            .iter()
            .any(|r| r.may_have_data(w.name, w.local_lo, w.local_hi))
    }) {
        return Ok(());
    }

    // Scratch buffer: (chunk_len, tile width), pre-filled with `T::ZERO`.
    // Readers only write the positions they cover; the zero fill stands in
    // for everything else (gaps, uncovered contigs).
    let mut buf = Array2::<T>::from_elem((chunk_len, tile_width), T::ZERO);

    for w in &windows {
        let (local_lo, local_hi) = (w.local_lo, w.local_hi);
        for (local_column, reader) in forked.iter_mut().enumerate() {
            let dst = buf.slice_mut(ndarray::s![
                w.row_lo..w.row_hi,
                local_column..local_column + 1
            ]);
            reader
                .read_into(w.name, local_lo, local_hi, dst)
                .map_err(|e| {
                    PbzError::Metadata(format!(
                        "reader {} failed on {} [{local_lo},{local_hi}): {e}",
                        task.columns.start + local_column as u64,
                        w.name
                    ))
                })?;
        }
    }

    // Collapse the column axis for scalar tracks; cohort tracks keep both.
    if track.rank() == 1 {
        let rank1 = buf.remove_axis(Axis(1)).into_dyn();
        track.write_flat::<T>(gs, ge, rank1)?;
    } else {
        track.write_flat_columns::<T>(gs..ge, task.columns.clone(), buf.into_dyn())?;
    }

    let chunk_bytes = (chunk_len * tile_width * mem::size_of::<T>()) as u64;
    state
        .bytes_written
        .fetch_add(chunk_bytes, Ordering::Relaxed);
    state.tasks_completed.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = progress {
        p.tick(chunk_bytes);
    }
    Ok(())
}

/// Drive a single [`MultiValueReader`] into several scalar tracks in one decode
/// pass. All tracks must share one genome and one chunk grid; each region task
/// reads the source once and writes one chunk per track.
pub fn run_multi_pipeline<R: MultiValueReader>(
    tracks: &[&Track],
    reader: R,
    config: &Config,
) -> Result<Report> {
    if tracks.is_empty() {
        return Err(PbzError::Metadata("multi pipeline: no tracks".into()));
    }
    let dtypes = reader.columns().to_vec();
    if dtypes.len() != tracks.len() {
        return Err(PbzError::Metadata(format!(
            "multi pipeline: {} columns for {} tracks",
            dtypes.len(),
            tracks.len()
        )));
    }

    let ref_genome = tracks[0].genome();
    let ref_checksum = ref_genome.checksum();
    let total = tracks[0].total_len();
    let step = (tracks[0].chunk_size()? as u64).max(1);

    for (i, t) in tracks.iter().enumerate() {
        if t.rank() != 1 {
            return Err(PbzError::Metadata(format!(
                "multi pipeline: track {:?} is not scalar",
                t.name()
            )));
        }
        if t.dtype() != dtypes[i] {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but column {i} is {}",
                    t.name(),
                    t.dtype(),
                    dtypes[i]
                ),
            });
        }
        if t.genome().checksum() != ref_checksum {
            return Err(PbzError::Metadata(format!(
                "multi pipeline: track {:?} genome differs from track {:?}",
                t.name(),
                tracks[0].name()
            )));
        }
        if t.total_len() != total || (t.chunk_size()? as u64) != step {
            return Err(PbzError::Metadata(format!(
                "multi pipeline: track {:?} geometry differs",
                t.name()
            )));
        }
    }

    let n_contigs = ref_genome.iter().filter(|(_, c)| c.length > 0).count();
    let genome = Arc::clone(ref_genome);

    let mut tasks: Vec<ChunkTask> = Vec::new();
    let n_chunks = total.div_ceil(step);
    for i in 0..n_chunks {
        let start = i * step;
        let end = (start + step).min(total);
        tasks.push(ChunkTask {
            start,
            end,
            columns: 0..1,
        });
    }

    let workers = config.workers.max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>((workers * 2).max(1));
    let reader = Arc::new(reader);
    let dtypes = Arc::new(dtypes);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
        first_err: Mutex::new(None),
    });

    thread::scope(|scope| {
        for _ in 0..workers {
            let task_rx = task_rx.clone();
            let reader = Arc::clone(&reader);
            let genome = Arc::clone(&genome);
            let dtypes = Arc::clone(&dtypes);
            let state = Arc::clone(&state);
            let progress = config.progress.clone();
            // `tracks: &[&Track]` is Copy; every worker reads it immutably.
            scope.spawn(move || {
                let forked = match reader.fork() {
                    Ok(r) => r,
                    Err(e) => {
                        state.record_err(PbzError::Metadata(format!("reader fork failed: {e}")));
                        return;
                    }
                };
                while let Ok(task) = task_rx.recv() {
                    if state.has_err() {
                        continue;
                    }
                    if let Err(e) = process_task_multi(
                        tracks,
                        dtypes.as_ref(),
                        &forked,
                        &genome,
                        &task,
                        progress.as_deref(),
                        &state,
                    ) {
                        state.record_err(e);
                    }
                }
            });
        }

        for task in tasks {
            if state.has_err() {
                break;
            }
            if task_tx.send(task).is_err() {
                break;
            }
        }
        drop(task_tx);
    });

    if let Some(ref p) = config.progress {
        p.done();
    }
    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    Ok(Report {
        contigs_written: n_contigs,
        bytes_written: state.bytes_written.load(Ordering::Relaxed),
        tasks_completed: state.tasks_completed.load(Ordering::Relaxed),
        tasks_skipped: state.tasks_skipped.load(Ordering::Relaxed),
    })
}

/// Drive one [`MultiValueReader`] into a single rank-2 track whose columns are
/// the reader's fields, in one decode pass. Every field must share the track's
/// dtype; tasks partition the position axis at full column width, so no two
/// tasks touch the same chunk even when the width spans several column chunks.
pub fn run_wide_pipeline<R: MultiValueReader>(
    track: &Track,
    reader: R,
    config: &Config,
) -> Result<Report> {
    if track.rank() != 2 {
        return Err(PbzError::Metadata(format!(
            "wide pipeline: track {:?} is not rank 2",
            track.name()
        )));
    }
    let width = track.columns_count()?;
    let dtypes = reader.columns().to_vec();
    if dtypes.len() != width {
        return Err(PbzError::Metadata(format!(
            "wide pipeline: {} fields for {width} columns",
            dtypes.len()
        )));
    }
    if dtypes.iter().any(|dtype| *dtype != track.dtype()) {
        return Err(PbzError::InvalidDtype {
            dtype: format!(
                "wide pipeline: track {:?} is {} but a field differs",
                track.name(),
                track.dtype()
            ),
        });
    }
    if reader.contigs().checksum() != track.genome().checksum() {
        return Err(PbzError::Metadata(
            "wide pipeline: reader genome differs from track".into(),
        ));
    }

    let genome = Arc::clone(track.genome());
    let n_contigs = genome.iter().filter(|(_, c)| c.length > 0).count();
    let total = track.total_len();
    let step = u64::try_from(track.write_unit_shape()?[0])
        .map_err(|_| PbzError::Metadata("position write unit exceeds u64".into()))?
        .max(1);
    let width_u64 =
        u64::try_from(width).map_err(|_| PbzError::Metadata("column count exceeds u64".into()))?;

    let mut tasks = Vec::new();
    for i in 0..total.div_ceil(step) {
        let start = i * step;
        tasks.push(ChunkTask {
            start,
            end: (start + step).min(total),
            columns: 0..width_u64,
        });
    }

    let workers = config.workers.max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>((workers * 2).max(1));
    let reader = Arc::new(reader);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
        first_err: Mutex::new(None),
    });

    thread::scope(|scope| {
        for _ in 0..workers {
            let task_rx = task_rx.clone();
            let reader = Arc::clone(&reader);
            let genome = Arc::clone(&genome);
            let state = Arc::clone(&state);
            let progress = config.progress.clone();
            scope.spawn(move || {
                let forked = match reader.fork() {
                    Ok(r) => r,
                    Err(e) => {
                        state.record_err(PbzError::Metadata(format!("reader fork failed: {e}")));
                        return;
                    }
                };
                while let Ok(task) = task_rx.recv() {
                    if state.has_err() {
                        continue;
                    }
                    if let Err(e) = process_wide_task(
                        track,
                        &forked,
                        &genome,
                        &task,
                        progress.as_deref(),
                        &state,
                    ) {
                        state.record_err(e);
                    }
                }
            });
        }
        for task in tasks {
            if state.has_err() || task_tx.send(task).is_err() {
                break;
            }
        }
        drop(task_tx);
    });

    if let Some(ref p) = config.progress {
        p.done();
    }
    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    Ok(Report {
        contigs_written: n_contigs,
        bytes_written: state.bytes_written.load(Ordering::Relaxed),
        tasks_completed: state.tasks_completed.load(Ordering::Relaxed),
        tasks_skipped: state.tasks_skipped.load(Ordering::Relaxed),
    })
}

fn process_wide_task<R: MultiValueReader>(
    track: &Track,
    reader: &R,
    genome: &crate::genome::Genome,
    task: &ChunkTask,
    progress: Option<&dyn ProgressSink>,
    state: &State,
) -> Result<()> {
    let rows = usize::try_from(task.end - task.start)
        .map_err(|_| PbzError::Metadata("wide tile row count exceeds usize".into()))?;
    let width = usize::try_from(task.columns.end)
        .map_err(|_| PbzError::Metadata("wide tile width exceeds usize".into()))?;
    let windows = contig_windows(genome, task.start, task.end);
    if skip_uncovered_task(&windows, state, |w| {
        reader.may_have_data(w.name, w.local_lo, w.local_hi)
    }) {
        return Ok(());
    }

    let mut buffer = MatrixBuffer::zeros(track.dtype(), rows, width)
        .map_err(|e| PbzError::Metadata(format!("alloc wide buffer: {e}")))?;

    for w in &windows {
        let (local_lo, local_hi) = (w.local_lo, w.local_hi);
        let mut sinks = buffer.sink_columns(w.row_lo..w.row_hi);
        reader
            .read_into(w.name, local_lo, local_hi, &mut sinks)
            .map_err(|e| {
                PbzError::Metadata(format!("wide read {} [{local_lo},{local_hi}): {e}", w.name))
            })?;
    }

    buffer.write_to_track(track, task.start..task.end, task.columns.clone())?;

    let bytes = rows as u64 * width as u64 * dtype_bytes(track.dtype()) as u64;
    state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    state.tasks_completed.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = progress {
        p.tick(bytes);
    }
    Ok(())
}

/// Drive several homogeneous [`MultiValueReader`] sources into one rank-2
/// track per reader field. Each task owns one physical position-by-column tile
/// across every target track.
pub fn run_matrix_pipeline<R: MultiValueReader>(
    tracks: &[&Track],
    readers: Vec<R>,
    config: &Config,
) -> Result<Report> {
    if tracks.is_empty() || readers.is_empty() {
        return Err(PbzError::Metadata(
            "matrix pipeline requires tracks and readers".into(),
        ));
    }
    let dtypes = readers[0].columns().to_vec();
    if dtypes.len() != tracks.len() {
        return Err(PbzError::Metadata(format!(
            "matrix pipeline: {} fields for {} tracks",
            dtypes.len(),
            tracks.len()
        )));
    }
    let reference_track = tracks[0];
    let reference_genome = reference_track.genome();
    let checksum = reference_genome.checksum();
    let total = reference_track.total_len();
    let write_unit = reference_track.write_unit_shape()?;
    if write_unit.len() != 2 {
        return Err(PbzError::Metadata(
            "matrix pipeline requires rank-2 tracks".into(),
        ));
    }
    let position_step = u64::try_from(write_unit[0])
        .map_err(|_| PbzError::Metadata("position write unit exceeds u64".into()))?;
    let column_step = u64::try_from(write_unit[1])
        .map_err(|_| PbzError::Metadata("column write unit exceeds u64".into()))?;
    let source_count = u64::try_from(readers.len())
        .map_err(|_| PbzError::Metadata("reader count exceeds u64".into()))?;

    for (field, track) in tracks.iter().enumerate() {
        if track.rank() != 2
            || track.columns_count()? as u64 != source_count
            || track.dtype() != dtypes[field]
            || track.genome().checksum() != checksum
            || track.total_len() != total
            || track.write_unit_shape()? != write_unit
        {
            return Err(PbzError::Metadata(format!(
                "matrix pipeline: track {:?} does not match the shared matrix layout",
                track.name()
            )));
        }
    }
    for reader in &readers {
        if reader.columns() != dtypes.as_slice() || reader.contigs().checksum() != checksum {
            return Err(PbzError::Metadata(
                "matrix pipeline: reader schema or genome differs".into(),
            ));
        }
    }

    let mut tasks = Vec::new();
    for position_index in 0..total.div_ceil(position_step) {
        let start = position_index * position_step;
        let end = (start + position_step).min(total);
        for column_start in (0..source_count).step_by(column_step as usize) {
            tasks.push(ChunkTask {
                start,
                end,
                columns: column_start..(column_start + column_step).min(source_count),
            });
        }
    }

    let workers = config.workers.max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>((workers * 2).max(1));
    let readers = Arc::new(readers);
    let dtypes = Arc::new(dtypes);
    let genome = Arc::clone(reference_genome);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
        first_err: Mutex::new(None),
    });

    thread::scope(|scope| {
        for _ in 0..workers {
            let task_rx = task_rx.clone();
            let readers = Arc::clone(&readers);
            let dtypes = Arc::clone(&dtypes);
            let genome = Arc::clone(&genome);
            let state = Arc::clone(&state);
            let progress = config.progress.clone();
            scope.spawn(move || {
                let mut cached_columns: Option<(std::ops::Range<usize>, Vec<R>)> = None;
                while let Ok(task) = task_rx.recv() {
                    if state.has_err() {
                        continue;
                    }
                    let Some((start, end)) = usize::try_from(task.columns.start)
                        .ok()
                        .zip(usize::try_from(task.columns.end).ok())
                    else {
                        state.record_err(PbzError::Metadata(
                            "task column range exceeds usize".into(),
                        ));
                        continue;
                    };
                    let columns = start..end;
                    if cached_columns
                        .as_ref()
                        .is_none_or(|(cached, _)| *cached != columns)
                    {
                        let forked = match readers[columns.clone()]
                            .iter()
                            .map(MultiValueReader::fork)
                            .collect::<crate::io::error::Result<Vec<_>>>()
                        {
                            Ok(forked) => forked,
                            Err(error) => {
                                state.record_err(PbzError::Metadata(format!(
                                    "reader fork failed: {error}"
                                )));
                                continue;
                            }
                        };
                        cached_columns = Some((columns, forked));
                    }
                    let Some((_, forked)) = cached_columns.as_ref() else {
                        continue;
                    };
                    if let Err(error) = process_matrix_task(
                        tracks,
                        dtypes.as_ref(),
                        forked,
                        &genome,
                        &task,
                        progress.as_deref(),
                        &state,
                    ) {
                        state.record_err(error);
                    }
                }
            });
        }
        for task in tasks {
            if state.has_err() || task_tx.send(task).is_err() {
                break;
            }
        }
        drop(task_tx);
    });

    if let Some(progress) = &config.progress {
        progress.done();
    }
    if let Some(error) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(error);
    }
    Ok(Report {
        contigs_written: reference_genome
            .iter()
            .filter(|(_, contig)| contig.length > 0)
            .count(),
        bytes_written: state.bytes_written.load(Ordering::Relaxed),
        tasks_completed: state.tasks_completed.load(Ordering::Relaxed),
        tasks_skipped: state.tasks_skipped.load(Ordering::Relaxed),
    })
}

#[allow(clippy::too_many_arguments)]
fn process_matrix_task<R: MultiValueReader>(
    tracks: &[&Track],
    dtypes: &[Dtype],
    readers: &[R],
    genome: &crate::genome::Genome,
    task: &ChunkTask,
    progress: Option<&dyn ProgressSink>,
    state: &State,
) -> Result<()> {
    let rows = usize::try_from(task.end - task.start)
        .map_err(|_| PbzError::Metadata("matrix tile row count exceeds usize".into()))?;
    let width = usize::try_from(task.columns.end - task.columns.start)
        .map_err(|_| PbzError::Metadata("matrix tile width exceeds usize".into()))?;
    let windows = contig_windows(genome, task.start, task.end);
    if skip_uncovered_task(&windows, state, |w| {
        readers
            .iter()
            .any(|r| r.may_have_data(w.name, w.local_lo, w.local_hi))
    }) {
        return Ok(());
    }

    let mut buffers = dtypes
        .iter()
        .map(|dtype| MatrixBuffer::zeros(*dtype, rows, width))
        .collect::<crate::io::error::Result<Vec<_>>>()
        .map_err(|error| PbzError::Metadata(format!("alloc matrix buffer: {error}")))?;
    for w in &windows {
        let (local_start, local_end) = (w.local_lo, w.local_hi);
        for (local_column, reader) in readers.iter().enumerate() {
            let mut sinks = buffers
                .iter_mut()
                .map(|buffer| buffer.sink_column_slice(w.row_lo..w.row_hi, local_column))
                .collect::<Vec<_>>();
            reader
                .read_into(w.name, local_start, local_end, &mut sinks)
                .map_err(|error| {
                    PbzError::Metadata(format!(
                        "matrix reader {} failed on {} [{local_start},{local_end}): {error}",
                        task.columns.start + local_column as u64,
                        w.name
                    ))
                })?;
        }
    }
    for (track, buffer) in tracks.iter().zip(buffers) {
        buffer.write_to_track(track, task.start..task.end, task.columns.clone())?;
    }
    let bytes = rows as u64
        * width as u64
        * dtypes
            .iter()
            .map(|dtype| dtype_bytes(*dtype) as u64)
            .sum::<u64>();
    state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
    state.tasks_completed.fetch_add(1, Ordering::Relaxed);
    if let Some(progress) = progress {
        progress.tick(bytes);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_task_multi<R: MultiValueReader>(
    tracks: &[&Track],
    dtypes: &[Dtype],
    reader: &R,
    genome: &crate::genome::Genome,
    task: &ChunkTask,
    progress: Option<&dyn ProgressSink>,
    state: &State,
) -> Result<()> {
    let (gs, ge) = (task.start, task.end);
    let chunk_len = (ge - gs) as usize;

    let windows = contig_windows(genome, gs, ge);
    if skip_uncovered_task(&windows, state, |w| {
        reader.may_have_data(w.name, w.local_lo, w.local_hi)
    }) {
        return Ok(());
    }

    let mut buffers: Vec<ColumnBuffer> = dtypes
        .iter()
        .map(|d| ColumnBuffer::zeros(*d, chunk_len))
        .collect::<crate::io::error::Result<Vec<_>>>()
        .map_err(|e| PbzError::Metadata(format!("alloc buffer: {e}")))?;

    for w in &windows {
        let (local_lo, local_hi) = (w.local_lo, w.local_hi);
        let mut sinks: Vec<ColumnSinkMut> = buffers
            .iter_mut()
            .map(|b| b.sink_slice(w.row_lo, w.row_hi))
            .collect();
        reader
            .read_into(w.name, local_lo, local_hi, &mut sinks)
            .map_err(|e| {
                PbzError::Metadata(format!(
                    "multi read {} [{local_lo},{local_hi}): {e}",
                    w.name
                ))
            })?;
    }

    let chunk_bytes: u64 =
        chunk_len as u64 * dtypes.iter().map(|d| dtype_bytes(*d) as u64).sum::<u64>();
    for (track, buffer) in tracks.iter().zip(buffers) {
        buffer.write_to_track(track, gs, ge)?;
    }

    state
        .bytes_written
        .fetch_add(chunk_bytes, Ordering::Relaxed);
    state.tasks_completed.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = progress {
        p.tick(chunk_bytes);
    }
    Ok(())
}

/// On-disk element width per multi-import dtype, for byte accounting.
fn dtype_bytes(d: Dtype) -> usize {
    match d {
        Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;
    use crate::genome::{Contig, Genome};
    use crate::io::{ColumnSinkMut, Dtype, MultiValueReader};
    use crate::track::TrackConfig;
    use crate::{PbzStore, Region};
    use ndarray::{Ix1, Ix2};
    use tempfile::TempDir;

    /// Fills every column with a constant cell string, exercising `fill_run`.
    /// `covered = false` makes `may_have_data` report no coverage anywhere
    /// and turns `read_into` into a hard failure, so a test using it proves
    /// the pipeline's skip path never reads (see
    /// `uncovered_tasks_are_skipped_and_store_no_chunks`).
    struct ConstMulti {
        genome: Genome,
        dtypes: Vec<Dtype>,
        cells: Vec<String>,
        covered: bool,
    }

    impl MultiValueReader for ConstMulti {
        fn contigs(&self) -> &Genome {
            &self.genome
        }
        fn columns(&self) -> &[Dtype] {
            &self.dtypes
        }
        fn may_have_data(&self, _contig: &str, _start: u64, _end: u64) -> bool {
            self.covered
        }
        fn read_into(
            &self,
            _contig: &str,
            start: u64,
            end: u64,
            sinks: &mut [ColumnSinkMut<'_>],
        ) -> crate::io::error::Result<()> {
            if !self.covered {
                return Err(crate::io::error::ReaderError::Other(anyhow::anyhow!(
                    "must not be called"
                )));
            }
            let len = (end - start) as usize;
            for (i, s) in sinks.iter_mut().enumerate() {
                s.fill_run(0, len, &self.cells[i])?;
            }
            Ok(())
        }
        fn fork(&self) -> crate::io::error::Result<Self> {
            Ok(ConstMulti {
                genome: self.genome.clone(),
                dtypes: self.dtypes.clone(),
                cells: self.cells.clone(),
                covered: self.covered,
            })
        }
    }

    #[test]
    fn multi_pipeline_writes_all_tracks_across_contig_boundary() {
        let dir = TempDir::new().unwrap();
        let g = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 50,
            },
            Contig {
                name: "chr2".into(),
                length: 30,
            },
        ])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("multi.pbz")).unwrap();
        store
            .create_track("a", g.clone(), TrackConfig::new(Dtype::I32).chunk_size(32))
            .unwrap();
        store
            .create_track("b", g.clone(), TrackConfig::new(Dtype::F32).chunk_size(32))
            .unwrap();

        let reader = ConstMulti {
            genome: g.clone(),
            dtypes: vec![Dtype::I32, Dtype::F32],
            cells: vec!["7".into(), "1.5".into()],
            covered: true,
        };
        let ta = store.track("a").unwrap();
        let tb = store.track("b").unwrap();
        let report = run_multi_pipeline(&[ta, tb], reader, &Config::default()).unwrap();
        assert_eq!(report.tasks_completed, 3); // ΣL=80, chunk 32 -> 32,32,16

        let ga = store.genome_for("a").unwrap();
        let r1 = Region {
            contig: ga.id("chr1").unwrap(),
            start: 0,
            end: 50,
        };
        let r2 = Region {
            contig: ga.id("chr2").unwrap(),
            start: 0,
            end: 30,
        };
        let a1 = store
            .track("a")
            .unwrap()
            .read_region::<i32>(&r1)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert!(a1.iter().all(|&v| v == 7));
        let b2 = store
            .track("b")
            .unwrap()
            .read_region::<f32>(&r2)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert!(b2.iter().all(|&v| v == 1.5));
    }

    #[test]
    fn uncovered_tasks_are_skipped_and_store_no_chunks() {
        let dir = TempDir::new().unwrap();
        let g = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 40,
        }])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("skip.pbz")).unwrap();
        store
            .create_track("a", g.clone(), TrackConfig::new(Dtype::I32).chunk_size(16))
            .unwrap();

        let reader = ConstMulti {
            genome: g.clone(),
            dtypes: vec![Dtype::I32],
            cells: vec!["7".into()],
            covered: false,
        };
        let ta = store.track("a").unwrap();
        let report = run_multi_pipeline(&[ta], reader, &Config::default()).unwrap();
        assert_eq!(report.tasks_completed, 0);
        // ΣL=40, chunk 16 -> 3 tasks (16, 16, 8), all skipped (never covered).
        assert_eq!(report.tasks_skipped, 3);

        let ga = store.genome_for("a").unwrap();
        let r = Region {
            contig: ga.id("chr1").unwrap(),
            start: 0,
            end: 40,
        };
        let vals = store
            .track("a")
            .unwrap()
            .read_region::<i32>(&r)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert!(vals.iter().all(|&v| v == 0));

        let mut entries: Vec<_> = std::fs::read_dir(dir.path().join("skip.pbz/a/values"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        entries.sort();
        assert_eq!(entries, vec![std::ffi::OsString::from("zarr.json")]);
    }

    #[test]
    fn matrix_pipeline_writes_two_fields_for_two_sources() {
        let dir = TempDir::new().unwrap();
        let genome = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 8,
        }])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("matrix.pbz")).unwrap();
        for (name, dtype) in [("coverage", Dtype::I32), ("score", Dtype::F32)] {
            store
                .create_track(
                    name,
                    genome.clone(),
                    TrackConfig::new(dtype)
                        .columns(vec!["a".into(), "b".into()])
                        .chunk_size(4)
                        .column_chunk_size(1),
                )
                .unwrap();
        }
        let readers = vec![
            ConstMulti {
                genome: genome.clone(),
                dtypes: vec![Dtype::I32, Dtype::F32],
                cells: vec!["5".into(), "1.5".into()],
                covered: true,
            },
            ConstMulti {
                genome: genome.clone(),
                dtypes: vec![Dtype::I32, Dtype::F32],
                cells: vec!["9".into(), "2.5".into()],
                covered: true,
            },
        ];
        run_matrix_pipeline(
            &[
                store.track("coverage").unwrap(),
                store.track("score").unwrap(),
            ],
            readers,
            &Config {
                workers: 2,
                ..Config::default()
            },
        )
        .unwrap();

        let region = Region {
            contig: genome.id("chr1").unwrap(),
            start: 0,
            end: 8,
        };
        let coverage = store
            .track("coverage")
            .unwrap()
            .read_region::<i32>(&region)
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let score = store
            .track("score")
            .unwrap()
            .read_region::<f32>(&region)
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert!(
            coverage
                .rows()
                .into_iter()
                .all(|row| row.to_vec() == vec![5, 9])
        );
        assert!(
            score
                .rows()
                .into_iter()
                .all(|row| row.to_vec() == vec![1.5, 2.5])
        );
    }
}
