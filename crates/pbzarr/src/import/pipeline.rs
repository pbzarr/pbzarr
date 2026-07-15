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
use crate::io::{Numeric, ValueReader};
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
    /// Optional progress observer.
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 4,
            chunk_size: None,
            column_chunk_size: None,
            progress: None,
        }
    }
}

/// Summary returned by `run_pipeline` on success.
pub struct Report {
    pub contigs_written: usize,
    pub bytes_written: u64,
    /// Number of chunk tasks that completed successfully.
    pub tasks_completed: usize,
}

/// A chunk-aligned segment `[start, end)` of the flat position axis, at full
/// column width. Boundaries land on the track's physical chunk grid.
#[derive(Clone, Copy, Debug)]
struct ChunkTask {
    start: u64,
    end: u64,
}

struct State {
    bytes_written: AtomicU64,
    tasks_completed: AtomicUsize,
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

    // Step by the track's physical write unit so each task writes one whole
    // chunk (or shard) that no other task touches.
    let step = (track.chunk_size()? as u64).max(1);
    let genome = Arc::clone(track.genome());
    let total = track.total_len();
    let n_contigs = genome.iter().filter(|(_, c)| c.length > 0).count();

    let mut tasks: Vec<ChunkTask> = Vec::new();
    let n_chunks = total.div_ceil(step);
    for i in 0..n_chunks {
        let start = i * step;
        let end = (start + step).min(total);
        tasks.push(ChunkTask { start, end });
    }

    let workers = config.workers.max(1);
    let task_cap = (workers * 2).max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>(task_cap);

    let readers = Arc::new(readers);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
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
                // Per-thread fork of every reader. The mutex-wrapped readers
                // in the source are otherwise serialized across calls; fork
                // gives each worker its own decoder state.
                let mut forked: Vec<R> = match readers
                    .iter()
                    .map(|r| r.fork())
                    .collect::<crate::io::error::Result<Vec<_>>>()
                {
                    Ok(v) => v,
                    Err(e) => {
                        state.record_err(PbzError::Metadata(format!("reader fork failed: {e}")));
                        return;
                    }
                };

                while let Ok(task) = task_rx.recv() {
                    if state.has_err() {
                        // Drain remaining tasks to let the channel close cleanly.
                        continue;
                    }
                    if let Err(e) = process_task::<T, R>(
                        track,
                        &mut forked,
                        n_readers,
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
    })
}

fn process_task<T, R>(
    track: &Track,
    forked: &mut [R],
    n_readers: usize,
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

    // Scratch buffer: (chunk_len, n_readers). Pre-fill with `T::ZERO`; readers
    // overwrite every position they cover.
    let mut buf = Array2::<T>::from_elem((chunk_len, n_readers), T::ZERO);

    // The task may straddle contig boundaries; fill each overlapping contig's
    // slice of the buffer by name. Contigs are in offset order, so stop once
    // one starts past the task.
    let offsets = genome.offsets();
    for (i, contig) in genome.contigs().iter().enumerate() {
        let c_start = offsets[i] as u64;
        if c_start >= ge {
            break;
        }
        let c_end = offsets[i + 1] as u64;
        if c_end <= gs {
            continue;
        }
        let ov_start = gs.max(c_start);
        let ov_end = ge.min(c_end);
        let (buf_lo, buf_hi) = ((ov_start - gs) as usize, (ov_end - gs) as usize);
        let (local_lo, local_hi) = (ov_start - c_start, ov_end - c_start);
        for (col_idx, reader) in forked.iter_mut().enumerate() {
            let dst = buf.slice_mut(ndarray::s![buf_lo..buf_hi, col_idx..col_idx + 1]);
            reader
                .read_into(&contig.name, local_lo, local_hi, dst)
                .map_err(|e| {
                    PbzError::Metadata(format!(
                        "reader {col_idx} failed on {} [{local_lo},{local_hi}): {e}",
                        contig.name
                    ))
                })?;
        }
    }

    // Collapse the column axis for scalar tracks; cohort tracks keep both.
    if track.rank() == 1 {
        let rank1 = buf.remove_axis(Axis(1)).into_dyn();
        track.write_flat::<T>(gs, ge, rank1)?;
    } else {
        track.write_flat::<T>(gs, ge, buf.into_dyn())?;
    }

    let chunk_bytes = (chunk_len * n_readers * mem::size_of::<T>()) as u64;
    state
        .bytes_written
        .fetch_add(chunk_bytes, Ordering::Relaxed);
    state.tasks_completed.fetch_add(1, Ordering::Relaxed);
    if let Some(p) = progress {
        p.tick(chunk_bytes);
    }
    Ok(())
}
