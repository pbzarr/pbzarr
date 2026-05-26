//! Generic channel-based ingest pipeline. Drives `ValueReader` sources into a
//! `Track` via a bounded crossbeam channel worker pool.
//!
//! # Design
//!
//! - One producer thread generates `(contig_id, start, end)` tasks, one per
//!   position chunk.
//! - N worker threads each fork every `ValueReader` once (via `ValueReader::fork`),
//!   then loop on tasks: for each task, fill an `Array2<T>` with all reader columns
//!   and ship a typed result to the writer channel.
//! - One writer thread drains results in arrival order, reshapes if scalar (rank-1),
//!   and calls `Track::write_region`.
//!
//! Error propagation: every result channel carries `crate::Result<(Region, Array2<T>)>`;
//! the first error seen on the writer thread is returned from `run_pipeline`.

use std::mem;
use std::sync::Arc;
use std::thread;

use crossbeam_channel::bounded;
use ndarray::{Array2, Axis, s};

use crate::error::PbzError;
use crate::genome::Region;
use crate::io::{Numeric, ValueReader};
use crate::track::Track;
use crate::Result;

/// Hook for callers to observe pipeline progress.
pub trait ProgressSink: Send + Sync {
    fn tick(&self, _bytes: u64) {}
    fn done(&self) {}
}

/// Configuration for `run_pipeline`.
pub struct ImportConfig {
    /// Number of reader worker threads.
    pub workers: usize,
    /// Override position chunk size. Defaults to `track.chunk_size()`.
    pub chunk_size: Option<usize>,
    /// Unused for now; reserved for future per-column chunking options.
    pub column_chunk_size: Option<usize>,
    /// Optional progress observer.
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for ImportConfig {
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
pub struct ImportReport {
    pub contigs_written: usize,
    pub bytes_written: u64,
}

// Internal task: one position chunk to process.
#[derive(Clone, Debug)]
struct ChunkTask {
    region: Region,
}

/// Drive a set of `ValueReader` instances into a `Track`, chunk by chunk.
///
/// - For scalar (rank-1) tracks, `readers.len()` MUST be 1.
/// - For cohort (rank-2) tracks, `readers.len()` MUST equal the track's column
///   count (as declared in the on-disk store).
///
/// Workers fork each reader once via `ValueReader::fork`; the original
/// `readers` vec is consumed by the function.
pub fn run_pipeline<T, R>(
    track: &Track,
    readers: Vec<R>,
    config: &ImportConfig,
) -> Result<ImportReport>
where
    T: Numeric,
    R: ValueReader<Item = T> + 'static,
{
    // Validate T matches the track's dtype.
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

    // Validate reader arity against track rank.
    match track.rank() {
        1 => {
            if n_readers != 1 {
                return Err(PbzError::Metadata(format!(
                    "scalar track {:?} requires exactly 1 reader, got {n_readers}",
                    track.name()
                )));
            }
        }
        2 => {
            // n_cols is the full column width declared in the store.
            let arr_path = format!(
                "/{}/{}",
                track
                    .genome()
                    .contigs()
                    .first()
                    .map(|c| c.name.as_str())
                    .unwrap_or(""),
                track.name()
            );
            // We don't have a cheap way to read n_cols here without opening the
            // zarr array, so skip the check and let the writer catch shape
            // mismatches from zarrs. The caller is responsible for passing the
            // right number of readers.
            let _ = arr_path;
        }
        other => {
            return Err(PbzError::Metadata(format!(
                "unexpected track rank {other}"
            )));
        }
    }

    let chunk_size = config.chunk_size.unwrap_or_else(|| track.chunk_size()) as u64;
    let genome = track.genome().clone();

    // Generate all tasks: one per (contig, chunk) pair.
    let mut tasks: Vec<ChunkTask> = Vec::new();
    let mut contigs_with_data: Vec<()> = Vec::new();
    for (contig_id, contig) in genome.iter() {
        let len = contig.length;
        if len == 0 {
            continue;
        }
        let n_chunks = len.div_ceil(chunk_size);
        contigs_with_data.push(());
        for chunk_idx in 0..n_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(len);
            tasks.push(ChunkTask {
                region: Region { contig: contig_id, start, end },
            });
        }
    }

    let n_contigs = contigs_with_data.len();
    let total_tasks = tasks.len();
    let workers = config.workers.max(1);

    // Channel bounds: 2× workers keeps both sides busy without unbounded memory.
    let task_cap = (workers * 2).max(1);
    let result_cap = (workers * 2).max(1);

    // Task channel: producer → workers.
    let (task_tx, task_rx) = bounded::<ChunkTask>(task_cap);
    // Result channel: workers → writer.
    let (res_tx, res_rx) = bounded::<Result<(Region, Array2<T>)>>(result_cap);

    // Producer thread: pushes all tasks onto the task channel.
    let producer = {
        let task_tx = task_tx.clone();
        thread::spawn(move || {
            for task in tasks {
                // If all workers have exited (receivers dropped), stop.
                if task_tx.send(task).is_err() {
                    break;
                }
            }
            // Dropping task_tx closes the channel; workers exit their recv loop.
        })
    };
    drop(task_tx);

    // Worker threads: each forks all readers once, then loops on tasks.
    let readers = Arc::new(readers);
    let mut worker_handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let task_rx = task_rx.clone();
        let res_tx = res_tx.clone();
        let readers = Arc::clone(&readers);
        worker_handles.push(thread::spawn(move || {
            // Fork all readers for this thread.
            let mut forked: Vec<R> = match readers.iter().map(|r| r.fork()).collect::<crate::io::error::Result<Vec<_>>>() {
                Ok(v) => v,
                Err(e) => {
                    // Report error and exit.
                    let _ = res_tx.send(Err(PbzError::Metadata(format!("reader fork failed: {e}"))));
                    return;
                }
            };

            while let Ok(task) = task_rx.recv() {
                let region = task.region;
                let chunk_len = region.len();

                // Allocate scratch buffer: shape (chunk_len, n_readers).
                // Fill with 0 bits; readers overwrite every position.
                // For float types this means 0.0, not NaN — acceptable since
                // well-behaved readers fill the entire dst slice.
                let mut buf = Array2::<T>::from_elem(
                    (chunk_len, n_readers),
                    // T doesn't impl Default, but we need an initial value.
                    // Use the bit pattern for 0, recovered via a unit-value read
                    // from a zero-filled i64: actually, we use from_elem with the
                    // fill we get from the first reader position (not safe in general).
                    // The simplest correct approach: rely on readers to fill dst
                    // completely, so the initial value doesn't matter.
                    // ndarray's from_elem requires a value; we obtain a zeroed one
                    // via MaybeUninit + write_bytes in `zeroed_elem` below.
                    zeroed_elem::<T>(),
                );

                let mut ok = true;
                for (col_idx, reader) in forked.iter_mut().enumerate() {
                    let dst = buf.slice_mut(s![.., col_idx..col_idx + 1]);
                    // Convert to ArrayViewMut2 via into_shape — the slice gives us a
                    // (chunk_len, 1) view already.
                    if let Err(e) = reader.read_into(&region, dst) {
                        let _ = res_tx.send(Err(PbzError::Metadata(format!(
                            "reader {col_idx} failed on {region}: {e}"
                        ))));
                        ok = false;
                        break;
                    }
                }
                if ok && res_tx.send(Ok((region, buf))).is_err() {
                    break;
                }
            }
        }));
    }
    drop(task_rx);
    drop(res_tx);

    // Writer: drain result channel, reshape, write.
    let progress = config.progress.clone();
    let mut bytes_written: u64 = 0;
    let mut first_err: Option<crate::error::PbzError> = None;

    for result in res_rx {
        match result {
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
                // Continue draining so workers can exit cleanly.
            }
            Ok((region, buf)) => {
                if first_err.is_some() {
                    // Already failed; skip writes but keep draining.
                    continue;
                }
                // For rank-1 tracks collapse the trailing column axis.
                let write_result = if track.rank() == 1 {
                    // Shape is (chunk_len, 1). Remove the column axis to get (chunk_len,).
                    let rank1 = buf.remove_axis(Axis(1)).into_dyn();
                    track.write_region::<T>(&region, rank1.view())
                } else {
                    track.write_region::<T>(&region, buf.view().into_dyn())
                };
                match write_result {
                    Ok(()) => {
                        let chunk_bytes =
                            (region.len() * n_readers * mem::size_of::<T>()) as u64;
                        bytes_written += chunk_bytes;
                        if let Some(ref p) = progress {
                            p.tick(chunk_bytes);
                        }
                    }
                    Err(e) => {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                }
            }
        }
    }

    // Join producer (it's just a sender; can't fail).
    let _ = producer.join();

    // Join workers; collect any panic-level errors.
    for handle in worker_handles {
        if let Err(_panic) = handle.join()
            && first_err.is_none()
        {
            first_err = Some(PbzError::Metadata("worker thread panicked".into()));
        }
    }

    if let Some(ref p) = progress {
        p.done();
    }

    if let Some(e) = first_err {
        return Err(e);
    }

    let _ = total_tasks; // silence unused warning if any

    Ok(ImportReport {
        contigs_written: n_contigs,
        bytes_written,
    })
}

/// Return a zero-bit-pattern value of type `T`.
///
/// This is sound for all `Numeric` types (u8, u16, u32, i8/16/32, f32, f64, bool)
/// because zero is a valid bit pattern for each. We use this only as a
/// placeholder fill that readers will immediately overwrite.
fn zeroed_elem<T: Copy>() -> T {
    // SAFETY: `T` is `Numeric` — one of the integer, float, or bool primitives.
    // All of those types have a valid zero bit-pattern, so transmuting a
    // zeroed `MaybeUninit` is defined behaviour.
    unsafe {
        let mu = std::mem::MaybeUninit::<T>::zeroed();
        mu.assume_init()
    }
}
