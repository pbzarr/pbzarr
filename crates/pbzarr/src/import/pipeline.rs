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
use crate::io::{ColumnBuffer, ColumnSinkMut, Dtype, MultiValueReader, Numeric, ValueReader};
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
        if !matches!(dtypes[i], Dtype::I32 | Dtype::F32 | Dtype::Bool) {
            return Err(PbzError::InvalidDtype {
                dtype: format!("multi import unsupported dtype {}", dtypes[i]),
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
        tasks.push(ChunkTask { start, end });
    }

    let workers = config.workers.max(1);
    let (task_tx, task_rx) = bounded::<ChunkTask>((workers * 2).max(1));
    let reader = Arc::new(reader);
    let dtypes = Arc::new(dtypes);
    let state = Arc::new(State {
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
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
    })
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

    let mut buffers: Vec<ColumnBuffer> = dtypes
        .iter()
        .map(|d| ColumnBuffer::zeros(*d, chunk_len))
        .collect::<crate::io::error::Result<Vec<_>>>()
        .map_err(|e| PbzError::Metadata(format!("alloc buffer: {e}")))?;

    // Fill each overlapping contig's slice of the buffers by name. Contigs are
    // in offset order, so stop once one starts past the task.
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
        let mut sinks: Vec<ColumnSinkMut> = buffers
            .iter_mut()
            .map(|b| b.sink_slice(buf_lo, buf_hi))
            .collect();
        reader
            .read_into(&contig.name, local_lo, local_hi, &mut sinks)
            .map_err(|e| {
                PbzError::Metadata(format!(
                    "multi read {} [{local_lo},{local_hi}): {e}",
                    contig.name
                ))
            })?;
    }

    let chunk_bytes: u64 =
        chunk_len as u64 * dtypes.iter().map(|d| dtype_bytes(*d) as u64).sum::<u64>();
    for (i, buf) in buffers.into_iter().enumerate() {
        match buf {
            ColumnBuffer::I32(a) => tracks[i].write_flat::<i32>(gs, ge, a.into_dyn())?,
            ColumnBuffer::F32(a) => tracks[i].write_flat::<f32>(gs, ge, a.into_dyn())?,
            ColumnBuffer::Bool(a) => tracks[i].write_flat::<bool>(gs, ge, a.into_dyn())?,
        }
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
        Dtype::Bool => 1,
        _ => 4, // I32 / F32
    }
}

#[cfg(test)]
mod multi_tests {
    use super::*;
    use crate::genome::{Contig, Genome};
    use crate::io::{ColumnSinkMut, Dtype, MultiValueReader};
    use crate::track::TrackConfig;
    use crate::{PbzStore, Region};
    use ndarray::Ix1;
    use tempfile::TempDir;

    /// Fills every column with a constant cell string, exercising `fill_run`.
    struct ConstMulti {
        genome: Genome,
        dtypes: Vec<Dtype>,
        cells: Vec<String>,
    }

    impl MultiValueReader for ConstMulti {
        fn contigs(&self) -> &Genome {
            &self.genome
        }
        fn columns(&self) -> &[Dtype] {
            &self.dtypes
        }
        fn read_into(
            &self,
            _contig: &str,
            start: u64,
            end: u64,
            sinks: &mut [ColumnSinkMut<'_>],
        ) -> crate::io::error::Result<()> {
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
}
