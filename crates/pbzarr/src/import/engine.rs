//! The import engine: one driver for every routing shape.
//!
//! A CHUNK is one inner-chunk-length window of the flat position axis and the
//! write unit; a SPAN is `decode_chunks` consecutive chunks, the window one
//! reader decodes in one pass; a PIECE is `(reader, span)`, the unit workers
//! pull from the channel; a BUFFER is one chunk of one target track
//! `(track, chunk, column chunk)`, pre-filled with that track's fill value.
//! A piece streams its span chunk by chunk, merging each chunk's slice into
//! the shared buffer as the walk crosses the boundary, so scratch stays one
//! chunk per active piece. The worker that finishes a buffer's last piece
//! writes it.
//!
//! Sharded tracks write through `store_array_subset_opt` with
//! `experimental_partial_encoding` on: for a subchunk-aligned subset the
//! partial encoder reads only the shard index, encodes the subchunks in
//! parallel, and APPENDS them instead of rewriting the shard. Each subchunk
//! must be written exactly once, and appends to one shard need a per-shard
//! mutex because every call rewrites that index. So a sharded chunk's buffers
//! flush together when its last buffer closes: one rectangular store call per
//! touched column shard.
//!
//! When no reader `may_have_data` over any of a buffer's pieces, the buffer is
//! never allocated. An unsharded buffer with no data elides its write; a
//! sharded chunk elides per column shard. An absent chunk already reads back
//! as the fill value.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, bounded};
use log::{debug, info, warn};
use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2, Axis, s};
use zarrs::array::{Array, ArrayShardedExt, ArraySubset, CodecOptions, FillValue};
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::Result;
use crate::error::PbzError;
use crate::genome::Genome;
use crate::import::config::{ProgressSink, Report};
use crate::import::estimate::human_bytes;
use crate::import::routing::{ImportRouting, SourceAxis, TrackTarget};
use crate::io::{Dtype, OutputSinkMut, ReaderError, ValueReader, WindowSink};
use crate::track::Track;

pub struct PipelineOptions {
    pub workers: usize,
    /// Open decode spans allowed at once. Bounds the open chunk buffers:
    /// each span holds up to `decode_chunks` buffers per track, and the
    /// merge scratch is bounded by the worker count instead. `0` = auto:
    /// `ceil(3 * workers / readers)`, so the runnable pieces target 3x the
    /// worker count, clamped to `[8, 256]` and to the span count.
    pub in_flight_spans: usize,
    /// Inner position chunks each reader decodes per piece. `0` = auto, see
    /// `auto_decode_chunks`; an explicit value is clamped to
    /// `[1, chunk count]`. Larger values cut per-piece fixed cost (one index
    /// seek per piece); buffers still close per chunk.
    pub decode_chunks: usize,
    /// Reader handles (forks) allowed across all readers at once. `0` = auto:
    /// derived from the soft fd limit with a margin for the store and
    /// indexes, see `auto_handle_budget`. Each reader's pool gets
    /// `budget / readers`, at least 1, at most `workers`.
    pub handle_budget: usize,
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            workers: 4,
            in_flight_spans: 0,
            decode_chunks: 0,
            handle_budget: 0,
            progress: None,
        }
    }
}

/// The auto rule for `PipelineOptions::in_flight_spans` = 0, before the span
/// count cap the engine applies.
pub fn auto_in_flight_spans(workers: usize, n_readers: usize) -> usize {
    (3 * workers.max(1))
        .div_ceil(n_readers.max(1))
        .clamp(8, 256)
}

/// Decode span cap in positions: 4M keeps a piece near one second of
/// decode on a deep BAM while bounding open chunk buffers per span.
const DECODE_SPAN_CAP: u64 = 4 << 20;

/// Auto rule for `PipelineOptions::decode_chunks` = 0: as many chunks as the
/// 4M cap allows, reduced so every worker sees about eight pieces.
pub fn auto_decode_chunks(total: u64, chunk_len: u64, n_readers: usize, workers: usize) -> usize {
    let chunk_len = chunk_len.max(1);
    let n_chunks = total.div_ceil(chunk_len).max(1);
    let cap = (DECODE_SPAN_CAP / chunk_len).max(1);
    let balance = total * n_readers.max(1) as u64 / (8 * workers.max(1) as u64 * chunk_len);
    usize::try_from(balance.clamp(1, cap).min(n_chunks)).unwrap_or(usize::MAX)
}

/// The auto rule for `PipelineOptions::handle_budget` = 0. Two fds per handle
/// (data + index); a quarter of the limit plus 128 fds stay free for the
/// store, shard files, and the base readers.
pub fn auto_handle_budget(soft_fd_limit: u64, n_readers: usize) -> usize {
    let usable = (soft_fd_limit * 3 / 4).saturating_sub(128) / 2;
    usize::try_from(usable)
        .unwrap_or(usize::MAX)
        .max(n_readers.max(1))
}

/// The process soft `RLIMIT_NOFILE`, capped so an unlimited rlimit does not
/// blow up the budget arithmetic.
#[cfg(unix)]
pub(crate) fn soft_fd_limit() -> u64 {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit writes into the provided struct and has no other effect.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) } == 0 {
        // `rlim_t` is not u64 on every unix target.
        #[allow(clippy::useless_conversion)]
        let current = u64::try_from(lim.rlim_cur).unwrap_or(u64::MAX);
        current.min(1 << 20)
    } else {
        1024
    }
}

#[cfg(not(unix))]
pub(crate) fn soft_fd_limit() -> u64 {
    1024
}

/// When attached, spans are opened in descending order of summed cost
/// (longest-first) instead of genome order.
pub trait CostModel: Send + Sync {
    fn span_cost(&self, contig: &str, start: u64, end: u64) -> u64;
}

/// One message per closed buffer, emitted on the optional tap channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TapMessage {
    Filled {
        track: String,
        position: Range<u64>,
        columns: Range<u64>,
    },
    /// The write was elided: no reader covered any of the buffer's pieces.
    Skipped {
        track: String,
        position: Range<u64>,
        columns: Range<u64>,
    },
}

fn dtype_bytes(d: Dtype) -> u64 {
    match d {
        Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

trait FillDecode: Sized {
    fn decode(fill: &FillValue) -> Result<Self>;
}

macro_rules! fill_decode_num {
    ($( $ty:ty ),* $(,)?) => { $(
        impl FillDecode for $ty {
            fn decode(fill: &FillValue) -> Result<Self> {
                let bytes: [u8; std::mem::size_of::<$ty>()] =
                    fill.as_ne_bytes().try_into().map_err(|_| {
                        PbzError::Metadata(format!(
                            "track fill value has {} byte(s); expected {} for {}",
                            fill.as_ne_bytes().len(),
                            std::mem::size_of::<$ty>(),
                            stringify!($ty),
                        ))
                    })?;
                Ok(<$ty>::from_ne_bytes(bytes))
            }
        }
    )* };
}

fill_decode_num!(u8, u16, u32, i8, i16, i32, f32, f64);

impl FillDecode for bool {
    fn decode(fill: &FillValue) -> Result<Self> {
        Ok(fill.as_ne_bytes().first().copied().unwrap_or(0) != 0)
    }
}

/// Disjoint rank-1 column views borrowing the original lifetime. `cols` must
/// be strictly increasing: each split consumes the view up to that column.
fn take_columns<'a, T>(
    mut view: ArrayViewMut2<'a, T>,
    cols: &[usize],
) -> Vec<ArrayViewMut1<'a, T>> {
    let mut out = Vec::with_capacity(cols.len());
    let mut offset = 0usize;
    for &c in cols {
        let (_, rest) = view.split_at(Axis(1), c - offset);
        let (column, rest) = rest.split_at(Axis(1), 1);
        out.push(column.remove_axis(Axis(1)));
        view = rest;
        offset = c + 1;
    }
    out
}

macro_rules! tile_buffer {
    ($( $variant:ident => $ty:ty ),* $(,)?) => {
        /// One buffer's dtype-erased `(rows, columns)` scratch, pre-filled with
        /// the target track's declared fill value.
        enum TileBuffer {
            $( $variant(Array2<$ty>), )*
        }

        impl TileBuffer {
            fn filled(dtype: Dtype, rows: usize, cols: usize, fill: &FillValue) -> Result<Self> {
                Ok(match dtype {
                    $( Dtype::$variant => TileBuffer::$variant(Array2::from_elem(
                        (rows, cols),
                        <$ty as FillDecode>::decode(fill)?,
                    )), )*
                })
            }

            /// `cols` holds strictly increasing buffer-local column indices;
            /// the sinks come back in `cols` order.
            fn column_sinks(&mut self, rows: Range<usize>, cols: &[usize]) -> Vec<OutputSinkMut<'_>> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        take_columns(a.slice_mut(s![rows.clone(), ..]), cols)
                            .into_iter()
                            .map(OutputSinkMut::$variant)
                            .collect()
                    } )*
                }
            }

            /// Overwrites every element with the track's fill value, so one
            /// allocation serves window after window.
            fn refill(&mut self, fill: &FillValue) -> Result<()> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        a.fill(<$ty as FillDecode>::decode(fill)?);
                        Ok(())
                    } )*
                }
            }

            /// Copies `src[src_rows, ..]` into `self[dst_rows, cols]`.
            fn copy_rows_from(
                &mut self,
                src: &TileBuffer,
                src_rows: Range<usize>,
                dst_rows: Range<usize>,
                cols: Range<usize>,
            ) -> Result<()> {
                match (self, src) {
                    $( (TileBuffer::$variant(dst), TileBuffer::$variant(src)) => {
                        dst.slice_mut(s![dst_rows, cols])
                            .assign(&src.slice(s![src_rows, ..]));
                        Ok(())
                    } )*
                    _ => Err(PbzError::Metadata(
                        "engine invariant: merge dtype mismatch".into(),
                    )),
                }
            }

            /// The buffer must be one chunk-aligned write unit.
            fn write_unsharded(self, track: &Track, pos: Range<u64>, cols: Range<u64>) -> Result<()> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        if track.rank() == 1 {
                            track.write_flat(pos.start, pos.end, a.remove_axis(Axis(1)).into_dyn())
                        } else {
                            track.write_flat_columns(pos, cols, a.into_dyn())
                        }
                    } )*
                }
            }

            /// Appends one subchunk. The caller must enable partial encoding in
            /// `opts` and hold that shard's mutex.
            fn store_subset(
                self,
                array: &Array<dyn ReadableWritableListableStorageTraits>,
                track_name: &str,
                rank: usize,
                pos: Range<u64>,
                cols: Range<u64>,
                opts: &CodecOptions,
            ) -> Result<()> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        let result = if rank == 1 {
                            #[allow(clippy::single_range_in_vec_init)]
                            let subset = ArraySubset::new_with_ranges(&[pos]);
                            array.store_array_subset_opt(&subset, a.remove_axis(Axis(1)).into_dyn(), opts)
                        } else {
                            let subset = ArraySubset::new_with_ranges(&[pos, cols]);
                            array.store_array_subset_opt(&subset, a.into_dyn(), opts)
                        };
                        result.map_err(|e| PbzError::Store(format!("write {track_name}: {e}")))
                    } )*
                }
            }
        }
    };
}

tile_buffer! {
    U8 => u8,
    U16 => u16,
    U32 => u32,
    I8 => i8,
    I16 => i16,
    I32 => i32,
    F32 => f32,
    F64 => f64,
    Bool => bool,
}

/// Resolved write geometry for one target track.
struct TargetGeom<'t> {
    track: &'t Track,
    array: Arc<Array<dyn ReadableWritableListableStorageTraits>>,
    rank: usize,
    dtype: Dtype,
    /// Track column count; 1 for rank-1 tracks.
    width: u64,
    /// Inner column chunk width; 1 for rank-1 tracks.
    col_chunk: u64,
    sharded: bool,
    /// Outer write-unit (shard) shape, for the per-shard mutex key.
    shard_pos: u64,
    shard_col: u64,
    fill: FillValue,
    /// Destination column range from the routing target.
    columns: Range<u64>,
    /// Inner position chunk length (the subchunk length when sharded): the
    /// engine's buffer height.
    array_chunk_len: u64,
}

impl TargetGeom<'_> {
    fn buffer_columns(&self, cc: u64) -> Range<u64> {
        if self.rank == 1 {
            0..1
        } else {
            let lo = cc * self.col_chunk;
            lo..(lo + self.col_chunk).min(self.width)
        }
    }
}

/// One contig's overlap with a decode span.
struct ContigWindow<'g> {
    name: &'g str,
    row_lo: usize,
    row_hi: usize,
    local_lo: u64,
    local_hi: u64,
}

fn contig_windows(genome: &Genome, start: u64, end: u64) -> Vec<ContigWindow<'_>> {
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

/// The producer acquires one slot per span before sending its pieces; the
/// worker that finishes a span's last piece releases it. `abort` unblocks the
/// producer on first error.
struct SpanGate {
    state: Mutex<(usize, bool)>,
    cv: Condvar,
    cap: usize,
}

impl SpanGate {
    fn new(cap: usize) -> Self {
        Self {
            state: Mutex::new((0, false)),
            cv: Condvar::new(),
            cap: cap.max(1),
        }
    }

    fn acquire(&self) {
        let mut s = self.state.lock().expect("span gate poisoned");
        while s.0 >= self.cap && !s.1 {
            s = self.cv.wait(s).expect("span gate poisoned");
        }
        if !s.1 {
            s.0 += 1;
        }
    }

    fn release(&self) {
        let mut s = self.state.lock().expect("span gate poisoned");
        s.0 = s.0.saturating_sub(1);
        drop(s);
        self.cv.notify_one();
    }

    fn abort(&self) {
        self.state.lock().expect("span gate poisoned").1 = true;
        self.cv.notify_all();
    }
}

/// One reader's work over one decode span: `[start, end)` covers
/// `decode_chunks` chunks, clamped at the end of the genome.
#[derive(Clone, Copy)]
struct Piece {
    reader: usize,
    span: usize,
    start: u64,
    end: u64,
}

/// One buffer a piece touches, plus the schema fields it routes there.
struct TouchEntry {
    t_idx: usize,
    cc: u64,
    /// `(schema field index, buffer-local column)`. The columns are
    /// consecutive and ascending; the merge copies them as one range.
    fields: Vec<(usize, usize)>,
}

type SlotKey = (usize, usize, u64); // (target, chunk, column chunk)

/// `(target, position shard, column shard)` -> its append-serializing mutex.
type ShardLockMap = HashMap<(usize, u64, u64), Arc<Mutex<()>>>;

struct SlotInner {
    /// Allocated lazily on the first covered piece; an elided buffer never
    /// allocates.
    data: Option<TileBuffer>,
    remaining: usize,
    any_data: bool,
}

struct SlotHandle {
    inner: Mutex<SlotInner>,
}

/// Closed buffers of one sharded `(target, chunk)`, held until the last one
/// arrives so the chunk flushes as one rectangle per column shard.
struct ChunkGroup {
    /// `(column chunk, buffer)`; an uncovered buffer stays `None`.
    parts: Vec<(u64, Option<TileBuffer>)>,
    remaining: usize,
}

/// Idle forks of one reader, capped at `capacity` live forks. Forks are
/// created on demand, so a reader no worker ever decodes opens nothing.
struct HandlePool<R> {
    /// `(idle handles, handles created)`.
    idle: Mutex<(Vec<R>, usize)>,
    cv: Condvar,
    capacity: usize,
}

impl<R: ValueReader> HandlePool<R> {
    fn new(capacity: usize) -> Self {
        Self {
            idle: Mutex::new((Vec::new(), 0)),
            cv: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    /// Blocks while every handle is checked out; returns `Ok(None)` once the
    /// engine has aborted so a waiting worker can drain.
    fn checkout(
        &self,
        fork: impl FnOnce() -> std::result::Result<R, crate::io::ReaderError>,
        aborted: &AtomicBool,
    ) -> Result<Option<R>> {
        let mut guard = self.idle.lock().expect("handle pool poisoned");
        loop {
            if let Some(handle) = guard.0.pop() {
                return Ok(Some(handle));
            }
            if guard.1 < self.capacity {
                guard.1 += 1;
                drop(guard);
                return match fork() {
                    Ok(handle) => Ok(Some(handle)),
                    Err(e) => {
                        self.idle.lock().expect("handle pool poisoned").1 -= 1;
                        self.cv.notify_one();
                        Err(PbzError::Reader(e))
                    }
                };
            }
            if aborted.load(Ordering::Relaxed) {
                return Ok(None);
            }
            // The timeout is what lets a blocked checkout notice an abort.
            guard = self
                .cv
                .wait_timeout(guard, Duration::from_millis(50))
                .expect("handle pool poisoned")
                .0;
        }
    }

    fn checkin(&self, handle: R) {
        self.idle
            .lock()
            .expect("handle pool poisoned")
            .0
            .push(handle);
        self.cv.notify_one();
    }
}

/// One stage's accumulated timing: sum, event count, and slowest event.
#[derive(Default)]
struct StageTimer {
    total_ns: AtomicU64,
    count: AtomicU64,
    max_ns: AtomicU64,
}

impl StageTimer {
    fn record(&self, elapsed: Duration) {
        let ns = elapsed.as_nanos() as u64;
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.max_ns.fetch_max(ns, Ordering::Relaxed);
    }

    fn total_secs(&self) -> f64 {
        self.total_ns.load(Ordering::Relaxed) as f64 / 1e9
    }

    fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    fn mean_secs(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            self.total_secs() / count as f64
        }
    }

    fn max_secs(&self) -> f64 {
        self.max_ns.load(Ordering::Relaxed) as f64 / 1e9
    }
}

#[derive(Default)]
struct RunTimings {
    /// Per piece: the coverage probe, the source read, and the merges and
    /// chunk closes the stream runs inside it.
    decode: StageTimer,
    /// Per piece: the `may_have_data` probe alone (also inside `decode`).
    probe: StageTimer,
    /// Per piece: time blocked checking a reader handle out of its pool.
    handle_wait: StageTimer,
    /// Per touched buffer: time blocked acquiring the slot's lock at merge.
    slot_wait: StageTimer,
    /// Per touched buffer: the region copy into the shared buffer, under
    /// the slot lock.
    merge: StageTimer,
    /// Per sharded flush segment: time blocked acquiring the shard's mutex.
    shard_wait: StageTimer,
    /// Per store call (encode + write), the shard mutex already held.
    write: StageTimer,
    worker_busy_ns: AtomicU64,
    worker_idle_ns: AtomicU64,
    /// Producer time blocked on the in-flight span gate.
    gate_wait_ns: AtomicU64,
}

fn ns_secs(counter: &AtomicU64) -> f64 {
    counter.load(Ordering::Relaxed) as f64 / 1e9
}

struct EngineState {
    slots: Mutex<HashMap<SlotKey, Arc<SlotHandle>>>,
    gate: SpanGate,
    span_remaining: Vec<AtomicUsize>,
    bytes_written: AtomicU64,
    tasks_completed: AtomicUsize,
    tasks_skipped: AtomicUsize,
    timings: RunTimings,
    err_flag: AtomicBool,
    first_err: Mutex<Option<PbzError>>,
}

impl EngineState {
    fn has_err(&self) -> bool {
        self.err_flag.load(Ordering::Relaxed)
    }

    fn record_err(&self, err: PbzError) {
        let mut slot = self.first_err.lock().expect("error slot poisoned");
        if slot.is_none() {
            debug!("import worker failed: {err}");
            *slot = Some(err);
        } else {
            warn!("additional import worker error (not reported): {err}");
        }
        drop(slot);
        self.err_flag.store(true, Ordering::Relaxed);
        self.gate.abort();
    }
}

/// Everything workers share.
struct RunCtx<'a> {
    state: &'a EngineState,
    geoms: &'a [TargetGeom<'a>],
    targets: &'a [TrackTarget],
    columns_from: Option<SourceAxis>,
    genome: &'a Genome,
    /// Per reader index: the buffers a piece for that reader touches.
    touch_plan: &'a [Vec<TouchEntry>],
    progress: Option<&'a dyn ProgressSink>,
    tap: Option<&'a Sender<TapMessage>>,
    partial_opts: CodecOptions,
    shard_locks: Mutex<ShardLockMap>,
    /// Sharded `(target, chunk)` groups awaiting their last buffer.
    chunk_groups: Mutex<HashMap<(usize, usize), ChunkGroup>>,
    /// Distinct column chunks per target: a sharded chunk group closes after
    /// this many buffers.
    group_sizes: Vec<usize>,
    /// Flat position count, so a piece can clamp its last chunk.
    total: u64,
}

impl RunCtx<'_> {
    fn slot_piece_count(&self, t_idx: usize, cc: u64) -> usize {
        match self.columns_from {
            Some(SourceAxis::Readers) => {
                let geom = &self.geoms[t_idx];
                let target = &self.targets[t_idx];
                let lo = (cc * geom.col_chunk).max(target.columns.start);
                let hi = ((cc + 1) * geom.col_chunk)
                    .min(geom.width)
                    .min(target.columns.end);
                (hi - lo) as usize
            }
            _ => 1,
        }
    }

    fn slot(&self, key: SlotKey) -> Arc<SlotHandle> {
        let mut map = self.state.slots.lock().expect("slot map poisoned");
        Arc::clone(map.entry(key).or_insert_with(|| {
            Arc::new(SlotHandle {
                inner: Mutex::new(SlotInner {
                    data: None,
                    remaining: self.slot_piece_count(key.0, key.2),
                    any_data: false,
                }),
            })
        }))
    }

    fn shard_lock(&self, t_idx: usize, pos: u64, col: u64) -> Arc<Mutex<()>> {
        let geom = &self.geoms[t_idx];
        let key = (t_idx, pos / geom.shard_pos, col / geom.shard_col.max(1));
        let mut map = self.shard_locks.lock().expect("shard lock map poisoned");
        Arc::clone(map.entry(key).or_default())
    }

    fn record_skipped(&self, geom: &TargetGeom<'_>, chunk: Range<u64>, columns: Range<u64>) {
        self.state.tasks_skipped.fetch_add(1, Ordering::Relaxed);
        if let Some(tap) = self.tap {
            // A dropped receiver detaches the tap; not an error.
            let _ = tap.send(TapMessage::Skipped {
                track: geom.track.name().to_owned(),
                position: chunk,
                columns,
            });
        }
    }

    fn record_written(&self, geom: &TargetGeom<'_>, chunk: Range<u64>, columns: Range<u64>) {
        let bytes =
            (chunk.end - chunk.start) * (columns.end - columns.start) * dtype_bytes(geom.dtype);
        self.state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.state.tasks_completed.fetch_add(1, Ordering::Relaxed);
        if let Some(p) = self.progress {
            p.tick(bytes);
        }
        if let Some(tap) = self.tap {
            let _ = tap.send(TapMessage::Filled {
                track: geom.track.name().to_owned(),
                position: chunk,
                columns,
            });
        }
    }

    /// Write, batch, or elide one closed buffer.
    fn close_buffer(
        &self,
        t_idx: usize,
        cc: u64,
        chunk_idx: usize,
        chunk: Range<u64>,
        data: Option<TileBuffer>,
        any_data: bool,
    ) -> Result<()> {
        if any_data && data.is_none() {
            return Err(PbzError::Metadata(
                "engine invariant: covered buffer has no data".into(),
            ));
        }
        let geom = &self.geoms[t_idx];
        if geom.sharded {
            return self.close_sharded(t_idx, chunk_idx, chunk, cc, data);
        }
        let columns = geom.buffer_columns(cc);
        let Some(data) = data else {
            self.record_skipped(geom, chunk, columns);
            return Ok(());
        };
        let write_start = Instant::now();
        data.write_unsharded(geom.track, chunk.clone(), columns.clone())?;
        self.state.timings.write.record(write_start.elapsed());
        self.record_written(geom, chunk, columns);
        Ok(())
    }

    /// Stash one closed sharded buffer in its `(target, chunk)` group; the
    /// group flushes when its last buffer arrives. Every append into a shard
    /// rewrites the shard index under that shard's mutex, so the batch gives
    /// one index rewrite and one lock acquisition per chunk instead of one per
    /// buffer.
    fn close_sharded(
        &self,
        t_idx: usize,
        chunk_idx: usize,
        chunk: Range<u64>,
        cc: u64,
        data: Option<TileBuffer>,
    ) -> Result<()> {
        let done = {
            let mut groups = self.chunk_groups.lock().expect("chunk group map poisoned");
            let group = groups
                .entry((t_idx, chunk_idx))
                .or_insert_with(|| ChunkGroup {
                    parts: Vec::with_capacity(self.group_sizes[t_idx]),
                    remaining: self.group_sizes[t_idx],
                });
            group.parts.push((cc, data));
            group.remaining -= 1;
            if group.remaining == 0 {
                groups.remove(&(t_idx, chunk_idx))
            } else {
                None
            }
        };
        match done {
            Some(group) => self.flush_chunk_group(t_idx, chunk, group),
            None => Ok(()),
        }
    }

    fn flush_chunk_group(&self, t_idx: usize, chunk: Range<u64>, group: ChunkGroup) -> Result<()> {
        let geom = &self.geoms[t_idx];
        let rows = (chunk.end - chunk.start) as usize;
        let shard_w = geom.shard_col.max(1);
        let mut parts = group.parts;
        parts.sort_unstable_by_key(|&(cc, _)| cc);

        // A chunk is one subchunk long and subchunks tile the shard, so the
        // chunk rectangle can only straddle shards on the column axis.
        let mut lo = 0;
        while lo < parts.len() {
            let seg_start = geom.buffer_columns(parts[lo].0).start;
            let shard_end = (seg_start / shard_w + 1) * shard_w;
            let mut hi = lo;
            let mut seg_end = seg_start;
            while hi < parts.len() && geom.buffer_columns(parts[hi].0).start < shard_end {
                seg_end = geom.buffer_columns(parts[hi].0).end;
                hi += 1;
            }
            let seg = seg_start..seg_end;

            // Elide only when the whole column shard has no data; one covered
            // buffer writes the full rectangle, fill-value columns included.
            // That trades a little elision for the batched write: fill
            // columns encode to almost nothing.
            if parts[lo..hi].iter().all(|(_, data)| data.is_none()) {
                for &(cc, _) in &parts[lo..hi] {
                    self.record_skipped(geom, chunk.clone(), geom.buffer_columns(cc));
                }
                lo = hi;
                continue;
            }

            let rect = if hi - lo == 1 {
                parts[lo]
                    .1
                    .take()
                    .expect("single covered part checked above")
            } else {
                let mut rect = TileBuffer::filled(
                    geom.dtype,
                    rows,
                    (seg.end - seg.start) as usize,
                    &geom.fill,
                )?;
                for (cc, data) in &mut parts[lo..hi] {
                    if let Some(data) = data.take() {
                        let cols = geom.buffer_columns(*cc);
                        let dst_lo = (cols.start - seg.start) as usize;
                        rect.copy_rows_from(
                            &data,
                            0..rows,
                            0..rows,
                            dst_lo..dst_lo + (cols.end - cols.start) as usize,
                        )?;
                    }
                }
                rect
            };

            let lock = self.shard_lock(t_idx, chunk.start, seg.start);
            let wait_start = Instant::now();
            let guard = lock.lock().expect("shard mutex poisoned");
            self.state.timings.shard_wait.record(wait_start.elapsed());
            let write_start = Instant::now();
            rect.store_subset(
                &geom.array,
                geom.track.name(),
                geom.rank,
                chunk.clone(),
                seg,
                &self.partial_opts,
            )?;
            self.state.timings.write.record(write_start.elapsed());
            drop(guard);

            for &(cc, _) in &parts[lo..hi] {
                self.record_written(geom, chunk.clone(), geom.buffer_columns(cc));
            }
            lo = hi;
        }
        Ok(())
    }
}

fn to_reader_err(err: PbzError) -> ReaderError {
    ReaderError::Other(anyhow::anyhow!("{err}"))
}

/// Per-piece streaming state: one chunk of scratch per touch entry, merged
/// into the shared buffers each time the walk leaves a chunk.
struct PieceSink<'a, 'c> {
    ctx: &'a RunCtx<'c>,
    piece: Piece,
    entries: &'a [TouchEntry],
    scratch: Vec<TileBuffer>,
    /// Flat and contig-local start of the contig window being streamed.
    window_flat_base: u64,
    window_local_base: u64,
    /// The chunk the last window belonged to, and whether any window of it
    /// was covered.
    current: Option<(u64, bool)>,
}

impl PieceSink<'_, '_> {
    fn chunk_len(&self) -> u64 {
        self.ctx.geoms[0].array_chunk_len.max(1)
    }

    fn flat(&self, local: u64) -> u64 {
        self.window_flat_base + (local - self.window_local_base)
    }

    /// Drop this piece's claim on every buffer of `chunk_idx` and write the
    /// ones it finished. Called exactly once per chunk of the span, covered
    /// or not, so the counts stay exact.
    fn finish_chunk(&self, chunk_idx: u64, covered: bool) -> Result<()> {
        let chunk_len = self.chunk_len();
        let lo = chunk_idx * chunk_len;
        let hi = (lo + chunk_len).min(self.ctx.total);
        let mut closed: Vec<(usize, u64, Option<TileBuffer>, bool)> = Vec::new();
        for entry in self.entries {
            let handle = self.ctx.slot((entry.t_idx, chunk_idx as usize, entry.cc));
            let mut guard = handle.inner.lock().expect("buffer slot poisoned");
            guard.remaining -= 1;
            guard.any_data |= covered;
            if guard.remaining == 0 {
                closed.push((entry.t_idx, entry.cc, guard.data.take(), guard.any_data));
            }
        }
        for (t_idx, cc, data, any_data) in closed {
            self.ctx
                .state
                .slots
                .lock()
                .expect("slot map poisoned")
                .remove(&(t_idx, chunk_idx as usize, cc));
            self.ctx
                .close_buffer(t_idx, cc, chunk_idx as usize, lo..hi, data, any_data)?;
        }
        Ok(())
    }

    /// Merge scratch rows `0..n` for the window at flat `flat_lo` into that
    /// chunk's buffers. Each lock covers one region copy, so pieces sharing a
    /// buffer stay concurrent through decode.
    fn merge(&self, flat_lo: u64, n: usize) -> Result<()> {
        let chunk_len = self.chunk_len();
        let chunk_idx = flat_lo / chunk_len;
        let chunk_lo = chunk_idx * chunk_len;
        let row0 = (flat_lo - chunk_lo) as usize;
        let rows = ((chunk_lo + chunk_len).min(self.ctx.total) - chunk_lo) as usize;
        for (idx, entry) in self.entries.iter().enumerate() {
            let wait_start = Instant::now();
            let handle = self.ctx.slot((entry.t_idx, chunk_idx as usize, entry.cc));
            let mut guard = handle.inner.lock().expect("buffer slot poisoned");
            self.ctx
                .state
                .timings
                .slot_wait
                .record(wait_start.elapsed());
            let merge_start = Instant::now();
            if guard.data.is_none() {
                let geom = &self.ctx.geoms[entry.t_idx];
                let columns = geom.buffer_columns(entry.cc);
                guard.data = Some(TileBuffer::filled(
                    geom.dtype,
                    rows,
                    (columns.end - columns.start) as usize,
                    &geom.fill,
                )?);
            }
            let dst_lo = entry.fields[0].1;
            guard
                .data
                .as_mut()
                .expect("buffer allocated above")
                .copy_rows_from(
                    &self.scratch[idx],
                    0..n,
                    row0..row0 + n,
                    dst_lo..dst_lo + entry.fields.len(),
                )?;
            self.ctx.state.timings.merge.record(merge_start.elapsed());
        }
        Ok(())
    }

    /// Finish every chunk of the span from `from_chunk` on with no data.
    fn finish_rest(&self, from_chunk: u64) -> Result<()> {
        let last = (self.piece.end - 1) / self.chunk_len();
        for chunk_idx in from_chunk..=last {
            self.finish_chunk(chunk_idx, false)?;
        }
        Ok(())
    }
}

impl WindowSink for PieceSink<'_, '_> {
    fn sinks(
        &mut self,
        start: u64,
        end: u64,
    ) -> std::result::Result<Vec<OutputSinkMut<'_>>, ReaderError> {
        let ctx = self.ctx;
        let entries = self.entries;
        let n = (end - start) as usize;
        let n_fields: usize = entries.iter().map(|e| e.fields.len()).sum();
        let mut staged: Vec<Option<OutputSinkMut<'_>>> = (0..n_fields).map(|_| None).collect();
        for (local, entry) in self.scratch.iter_mut().zip(entries) {
            local
                .refill(&ctx.geoms[entry.t_idx].fill)
                .map_err(to_reader_err)?;
            let cols: Vec<usize> = (0..entry.fields.len()).collect();
            for (&(field, _), sink) in entry.fields.iter().zip(local.column_sinks(0..n, &cols)) {
                staged[field] = Some(sink);
            }
        }
        Ok(staged
            .into_iter()
            .map(|s| s.expect("routing covers every schema field"))
            .collect())
    }

    fn done(
        &mut self,
        start: u64,
        end: u64,
        covered: bool,
    ) -> std::result::Result<(), ReaderError> {
        let flat_lo = self.flat(start);
        let chunk_idx = flat_lo / self.chunk_len();
        // Windows arrive in ascending order, so leaving a chunk means the
        // piece is done with it. A chunk split by a contig boundary keeps its
        // coverage flag across both windows.
        let carried = match self.current {
            Some((cur, cur_covered)) if cur == chunk_idx => cur_covered,
            Some((cur, cur_covered)) => {
                self.finish_chunk(cur, cur_covered).map_err(to_reader_err)?;
                false
            }
            None => false,
        };
        if covered {
            self.merge(flat_lo, (end - start) as usize)
                .map_err(to_reader_err)?;
        }
        self.current = Some((chunk_idx, carried | covered));
        Ok(())
    }
}

fn process_piece<R: ValueReader>(ctx: &RunCtx<'_>, reader: &mut R, piece: Piece) -> Result<()> {
    let windows = contig_windows(ctx.genome, piece.start, piece.end);
    let probe_start = Instant::now();
    let covered = windows
        .iter()
        .any(|w| reader.may_have_data(w.name, w.local_lo, w.local_hi));
    let mut decode = probe_start.elapsed();
    ctx.state.timings.probe.record(decode);

    let entries = &ctx.touch_plan[piece.reader];
    let chunk_len = ctx.geoms[0].array_chunk_len.max(1);
    let first_chunk = piece.start / chunk_len;
    let mut sink = PieceSink {
        ctx,
        piece,
        entries,
        scratch: Vec::new(),
        window_flat_base: 0,
        window_local_base: 0,
        current: None,
    };

    if covered {
        let read_start = Instant::now();
        // One chunk of scratch per touched entry, reused window after window.
        // Peak scratch memory gains one such set per active worker.
        for entry in entries {
            let geom = &ctx.geoms[entry.t_idx];
            sink.scratch.push(TileBuffer::filled(
                geom.dtype,
                chunk_len as usize,
                entry.fields.len(),
                &geom.fill,
            )?);
        }
        for w in &windows {
            let flat_lo = piece.start + w.row_lo as u64;
            let flat_hi = piece.start + w.row_hi as u64;
            sink.window_flat_base = flat_lo;
            sink.window_local_base = w.local_lo;
            let mut cuts = Vec::new();
            let mut boundary = (flat_lo / chunk_len + 1) * chunk_len;
            while boundary < flat_hi {
                cuts.push(w.local_lo + (boundary - flat_lo));
                boundary += chunk_len;
            }
            reader
                .read_windows(w.name, w.local_lo, w.local_hi, &cuts, &mut sink)
                .map_err(PbzError::Reader)?;
        }
        decode += read_start.elapsed();
    }
    ctx.state.timings.decode.record(decode);

    // Finish the last streamed chunk, then every chunk the stream never
    // reached, so an uncovered chunk inside a covered span still elides.
    let next = match sink.current.take() {
        Some((chunk_idx, chunk_covered)) => {
            sink.finish_chunk(chunk_idx, chunk_covered)?;
            chunk_idx + 1
        }
        None => first_chunk,
    };
    sink.finish_rest(next)
}

fn build_touch_plan(
    routing: &ImportRouting,
    geoms: &[TargetGeom<'_>],
    n_readers: usize,
    n_fields: usize,
) -> Vec<Vec<TouchEntry>> {
    let mut plan = Vec::with_capacity(n_readers);
    for reader in 0..n_readers {
        let mut entries: Vec<TouchEntry> = Vec::new();
        for (t_idx, target) in routing.targets.iter().enumerate() {
            let geom = &geoms[t_idx];
            match routing.columns_from {
                Some(SourceAxis::Readers) => {
                    let col = target.columns.start + reader as u64;
                    let cc = col / geom.col_chunk;
                    entries.push(TouchEntry {
                        t_idx,
                        cc,
                        fields: vec![(target.field, (col - cc * geom.col_chunk) as usize)],
                    });
                }
                Some(SourceAxis::Fields) => {
                    // Single reader. Columns ascend with the field index, so
                    // runs of one column chunk are always adjacent.
                    let mut by_chunk: Vec<(u64, Vec<(usize, usize)>)> = Vec::new();
                    for field in 0..n_fields {
                        let col = target.columns.start + field as u64;
                        let cc = col / geom.col_chunk;
                        let local = (col - cc * geom.col_chunk) as usize;
                        match by_chunk.last_mut() {
                            Some((last_cc, fields)) if *last_cc == cc => {
                                fields.push((field, local));
                            }
                            _ => by_chunk.push((cc, vec![(field, local)])),
                        }
                    }
                    for (cc, fields) in by_chunk {
                        entries.push(TouchEntry { t_idx, cc, fields });
                    }
                }
                None => entries.push(TouchEntry {
                    t_idx,
                    cc: 0,
                    fields: vec![(target.field, 0)],
                }),
            }
        }
        plan.push(entries);
    }
    plan
}

pub(crate) fn run_import<R: ValueReader>(
    readers: Vec<R>,
    tracks: &[&Track],
    routing: &ImportRouting,
    options: &PipelineOptions,
    cost_model: Option<&dyn CostModel>,
    tap: Option<&Sender<TapMessage>>,
) -> Result<Report> {
    let n_readers = readers.len();
    let n_fields = readers[0].output_schema().len();

    let mut geoms: Vec<TargetGeom<'_>> = Vec::with_capacity(tracks.len());
    for (track, target) in tracks.iter().zip(&routing.targets) {
        let array = track.values_array()?;
        let rank = track.rank();
        let width = track.columns_count()? as u64;
        let sharded = array.is_sharded();
        let write_unit = track.write_unit_shape()?;
        let (chunk_len, col_chunk) = if sharded {
            let sub = array.effective_subchunk_shape().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "track {:?}: sharded values array with indeterminate subchunk shape",
                    track.name()
                ))
            })?;
            let sub = sub.as_slice();
            let pos = sub[0].get();
            let col = if rank == 2 { sub[1].get() } else { 1 };
            (pos, col)
        } else {
            let pos = write_unit[0] as u64;
            let col = if rank == 2 { write_unit[1] as u64 } else { 1 };
            (pos, col)
        };
        // Buffers are whole inner chunks, so an unaligned destination range
        // would clobber the columns outside it.
        if target.columns.start % col_chunk != 0
            || (target.columns.end % col_chunk != 0 && target.columns.end != width)
        {
            return Err(PbzError::Metadata(format!(
                "import: target column range {:?} on track {:?} is not aligned to its \
                 column chunk width {col_chunk}",
                target.columns,
                track.name()
            )));
        }
        if let Some(previous) = geoms.first() {
            let prev_chunk = previous.array_chunk_len;
            if prev_chunk != chunk_len {
                return Err(PbzError::Metadata(format!(
                    "import: track {:?} inner position chunk {chunk_len} differs from \
                     track {:?} ({prev_chunk}); all targets must share the position grid",
                    track.name(),
                    previous.track.name()
                )));
            }
        }
        geoms.push(TargetGeom {
            track,
            fill: array.fill_value().clone(),
            rank,
            dtype: track.dtype(),
            width,
            col_chunk,
            sharded,
            shard_pos: write_unit[0] as u64,
            shard_col: if rank == 2 { write_unit[1] as u64 } else { 1 },
            columns: target.columns.clone(),
            array,
            array_chunk_len: chunk_len,
        });
    }

    let genome = Arc::clone(tracks[0].genome());
    let total = tracks[0].total_len();
    let workers = options.workers.max(1);
    let chunk_len = geoms[0].array_chunk_len.max(1);
    let n_chunks = usize::try_from(total.div_ceil(chunk_len))
        .map_err(|_| PbzError::Metadata("chunk count exceeds usize".into()))?;
    let decode_chunks = match options.decode_chunks {
        0 => auto_decode_chunks(total, chunk_len, n_readers, workers),
        n => n.clamp(1, n_chunks.max(1)),
    };
    let span_len = chunk_len * decode_chunks as u64;
    let n_spans = usize::try_from(total.div_ceil(span_len))
        .map_err(|_| PbzError::Metadata("span count exceeds usize".into()))?;

    let spans: Vec<(u64, u64)> = (0..n_spans as u64)
        .map(|i| (i * span_len, ((i + 1) * span_len).min(total)))
        .collect();

    // The gate and its counters key on the span INDEX, so the open order is
    // free to change.
    let mut span_order: Vec<usize> = (0..n_spans).collect();
    if let Some(model) = cost_model {
        let costs: Vec<u64> = spans
            .iter()
            .map(|&(s, e)| {
                contig_windows(&genome, s, e)
                    .iter()
                    .map(|w| model.span_cost(w.name, w.local_lo, w.local_hi))
                    .sum()
            })
            .collect();
        span_order.sort_by(|&a, &b| costs[b].cmp(&costs[a]));
    }

    if let Some(progress) = &options.progress {
        let total_bytes: u64 = geoms
            .iter()
            .map(|g| total * (g.columns.end - g.columns.start) * dtype_bytes(g.dtype))
            .sum();
        progress.set_total(total_bytes);
    }

    let touch_plan = build_touch_plan(routing, &geoms, n_readers, n_fields);

    let group_sizes: Vec<usize> = {
        let mut ccs: Vec<std::collections::BTreeSet<u64>> = vec![Default::default(); geoms.len()];
        for entries in &touch_plan {
            for entry in entries {
                ccs[entry.t_idx].insert(entry.cc);
            }
        }
        ccs.into_iter().map(|set| set.len()).collect()
    };

    let in_flight = match options.in_flight_spans {
        0 => auto_in_flight_spans(workers, n_readers).min(n_spans.max(1)),
        n => n,
    };
    let handle_budget = match options.handle_budget {
        0 => auto_handle_budget(soft_fd_limit(), n_readers),
        n => n,
    };
    let pool_size = (handle_budget / n_readers.max(1)).clamp(1, workers);
    let pieces = n_spans * n_readers;

    // Open buffers are bounded by the spans in flight; scratch by the
    // workers. Both are whole chunks.
    let chunk_row_bytes: u64 = geoms
        .iter()
        .map(|g| chunk_len * (g.columns.end - g.columns.start) * dtype_bytes(g.dtype))
        .sum();
    let scratch_chunk_bytes: u64 = touch_plan
        .iter()
        .map(|entries| {
            entries
                .iter()
                .map(|e| chunk_len * e.fields.len() as u64 * dtype_bytes(geoms[e.t_idx].dtype))
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0);
    let buffer_ceiling = (in_flight as u64)
        .saturating_mul(decode_chunks as u64)
        .saturating_mul(chunk_row_bytes)
        .saturating_add((workers as u64).saturating_mul(scratch_chunk_bytes));

    let started = Instant::now();
    let names: Vec<&str> = tracks.iter().map(|t| t.name()).collect();
    info!(
        "import pipeline: {} track(s) {names:?}, {total} positions, {n_chunks} chunk(s) of \
         {chunk_len}, {n_spans} span(s) of {decode_chunks} chunk(s), {n_readers} reader(s), \
         {pieces} piece(s), {workers} workers, {in_flight} span(s) in flight, {handle_budget} \
         handle(s) budget ({pool_size} per reader), buffer ceiling {} (reader scratch \
         excluded)",
        tracks.len(),
        human_bytes(buffer_ceiling)
    );

    let state = EngineState {
        slots: Mutex::new(HashMap::new()),
        gate: SpanGate::new(in_flight),
        span_remaining: (0..n_spans).map(|_| AtomicUsize::new(n_readers)).collect(),
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
        timings: RunTimings::default(),
        err_flag: AtomicBool::new(false),
        first_err: Mutex::new(None),
    };
    let mut partial_opts = CodecOptions::default();
    partial_opts.set_experimental_partial_encoding(true);
    let ctx = RunCtx {
        state: &state,
        geoms: &geoms,
        targets: &routing.targets,
        columns_from: routing.columns_from,
        genome: &genome,
        touch_plan: &touch_plan,
        progress: options.progress.as_deref(),
        tap,
        partial_opts,
        shard_locks: Mutex::new(HashMap::new()),
        chunk_groups: Mutex::new(HashMap::new()),
        group_sizes,
        total,
    };

    // `fork` takes `&self`, but `ValueReader` is not `Sync`, so the base
    // readers are reached under a lock held only across the fork call.
    let originals = Mutex::new(readers);
    let pools: Vec<HandlePool<R>> = (0..n_readers).map(|_| HandlePool::new(pool_size)).collect();

    let (piece_tx, piece_rx) = bounded::<Piece>((workers * 2).max(1));

    thread::scope(|scope| {
        for _ in 0..workers {
            let piece_rx = piece_rx.clone();
            let ctx = &ctx;
            let originals = &originals;
            let pools = &pools;
            scope.spawn(move || {
                let mut busy = Duration::ZERO;
                let mut idle = Duration::ZERO;
                loop {
                    let wait_start = Instant::now();
                    let Ok(piece) = piece_rx.recv() else { break };
                    idle += wait_start.elapsed();
                    let work_start = Instant::now();
                    // Stalling on a pool is not work, so it comes back out of
                    // `busy` below and is reported as its own stage.
                    let mut stalled = Duration::ZERO;
                    'piece: {
                        if ctx.state.has_err() {
                            // Drain so the channel closes cleanly; the gate is
                            // already aborted, so nothing blocks on us.
                            break 'piece;
                        }
                        let checkout_start = Instant::now();
                        let checked_out = pools[piece.reader].checkout(
                            || originals.lock().expect("readers poisoned")[piece.reader].fork(),
                            &ctx.state.err_flag,
                        );
                        stalled = checkout_start.elapsed();
                        ctx.state.timings.handle_wait.record(stalled);
                        let mut reader = match checked_out {
                            Ok(Some(reader)) => reader,
                            Ok(None) => break 'piece,
                            Err(e) => {
                                ctx.state.record_err(e);
                                break 'piece;
                            }
                        };
                        let result = process_piece(ctx, &mut reader, piece);
                        pools[piece.reader].checkin(reader);
                        if let Err(e) = result {
                            ctx.state.record_err(e);
                            break 'piece;
                        }
                        if ctx.state.span_remaining[piece.span].fetch_sub(1, Ordering::AcqRel) == 1
                        {
                            ctx.state.gate.release();
                        }
                    }
                    busy += work_start.elapsed().saturating_sub(stalled);
                }
                let timings = &ctx.state.timings;
                timings
                    .worker_busy_ns
                    .fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
                timings
                    .worker_idle_ns
                    .fetch_add(idle.as_nanos() as u64, Ordering::Relaxed);
            });
        }

        let mut gate_wait = Duration::ZERO;
        'produce: for &span_idx in &span_order {
            if state.has_err() {
                break;
            }
            let gate_start = Instant::now();
            state.gate.acquire();
            gate_wait += gate_start.elapsed();
            if state.has_err() {
                break;
            }
            let (start, end) = spans[span_idx];
            for reader in 0..n_readers {
                let piece = Piece {
                    reader,
                    span: span_idx,
                    start,
                    end,
                };
                if piece_tx.send(piece).is_err() {
                    break 'produce;
                }
            }
        }
        state
            .timings
            .gate_wait_ns
            .store(gate_wait.as_nanos() as u64, Ordering::Relaxed);
        drop(piece_tx);
    });

    if let Some(progress) = &options.progress {
        progress.done();
    }

    let completed = state.tasks_completed.load(Ordering::Relaxed);
    let skipped = state.tasks_skipped.load(Ordering::Relaxed);
    let bytes = state.bytes_written.load(Ordering::Relaxed);
    info!(
        "import pipeline finished: {completed} buffer(s) written, {skipped} elided, \
         {bytes} bytes in {:.1?}",
        started.elapsed()
    );
    if completed == 0 && skipped > 0 {
        warn!(
            "import pipeline wrote no data: all {skipped} buffers were elided because \
             no source covered them"
        );
    }

    let wall = started.elapsed().as_secs_f64();
    let timings = &state.timings;
    let busy = ns_secs(&timings.worker_busy_ns);
    let idle = ns_secs(&timings.worker_idle_ns);
    let gate_wait = ns_secs(&timings.gate_wait_ns);
    let busy_pct = if busy + idle > 0.0 {
        100.0 * busy / (busy + idle)
    } else {
        0.0
    };
    info!(
        "import timing: wall {wall:.1}s workers {workers}\n\
         decode (incl. merge+write): total {:.1}s  pieces {}  mean {:.2}s  max {:.2}s\n\
         probe  : total {:.1}s\n\
         slot wait: total {:.1}s  waits {}  mean {:.2}s  max {:.2}s\n\
         merge  : total {:.1}s  copies {}  mean {:.2}s  max {:.2}s\n\
         shard wait: total {:.1}s  waits {}  mean {:.2}s  max {:.2}s\n\
         write  : total {:.1}s  stores {}  mean {:.2}s  max {:.2}s\n\
         worker busy {busy:.0}s / idle {idle:.0}s ({busy_pct:.0}% busy)\n\
         gate wait: {gate_wait:.1}s\n\
         handle wait: total {:.1}s  waits {}  mean {:.2}s  max {:.2}s",
        timings.decode.total_secs(),
        timings.decode.count(),
        timings.decode.mean_secs(),
        timings.decode.max_secs(),
        timings.probe.total_secs(),
        timings.slot_wait.total_secs(),
        timings.slot_wait.count(),
        timings.slot_wait.mean_secs(),
        timings.slot_wait.max_secs(),
        timings.merge.total_secs(),
        timings.merge.count(),
        timings.merge.mean_secs(),
        timings.merge.max_secs(),
        timings.shard_wait.total_secs(),
        timings.shard_wait.count(),
        timings.shard_wait.mean_secs(),
        timings.shard_wait.max_secs(),
        timings.write.total_secs(),
        timings.write.count(),
        timings.write.mean_secs(),
        timings.write.max_secs(),
        timings.handle_wait.total_secs(),
        timings.handle_wait.count(),
        timings.handle_wait.mean_secs(),
        timings.handle_wait.max_secs(),
    );

    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    // Every piece drops its claim on every chunk of its span, so a clean run
    // ends with nothing open. A reader that skipped a window would otherwise
    // leave that chunk unwritten, uncounted, and unreported.
    let open = state.slots.lock().expect("slot map poisoned").len()
        + ctx
            .chunk_groups
            .lock()
            .expect("chunk group map poisoned")
            .len();
    if open > 0 {
        return Err(PbzError::Metadata(format!(
            "engine invariant: {open} chunk buffer(s) never closed; a reader did not \
             report every window of its span"
        )));
    }

    Ok(Report {
        contigs_written: genome.iter().filter(|(_, c)| c.length > 0).count(),
        pieces,
        bytes_written: bytes,
        tasks_completed: completed,
        tasks_skipped: skipped,
        wall_seconds: wall,
        decode_seconds: timings.decode.total_secs(),
        probe_seconds: timings.probe.total_secs(),
        write_seconds: timings.write.total_secs(),
        worker_busy_seconds: busy,
        worker_idle_seconds: idle,
        gate_wait_seconds: gate_wait,
        handle_wait_seconds: timings.handle_wait.total_secs(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use ndarray::{Ix1, Ix2};
    use tempfile::TempDir;

    use super::*;
    use crate::genome::{Contig, Region};
    use crate::import::Import;
    use crate::io::{OutputSchema, ValueReader};
    use crate::store::PbzStore;
    use crate::track::TrackConfig;

    /// Synthetic single-field reader over one contig: position `p` inside
    /// `covered` (or `extra`) reads as `base + p`; everything else is left
    /// untouched.
    #[derive(Clone)]
    struct SynthReader {
        genome: Genome,
        schema: OutputSchema,
        covered: Range<u64>,
        extra: Option<Range<u64>>,
        base: i64,
    }

    impl SynthReader {
        fn new(genome: Genome, dtype: Dtype, covered: Range<u64>, base: i64) -> Self {
            Self {
                genome,
                schema: OutputSchema::single("value", dtype),
                covered,
                extra: None,
                base,
            }
        }

        /// A second covered range, disjoint from the first.
        fn with_extra_cover(mut self, extra: Range<u64>) -> Self {
            self.extra = Some(extra);
            self
        }

        fn ranges(&self) -> impl Iterator<Item = &Range<u64>> {
            std::iter::once(&self.covered).chain(self.extra.iter())
        }
    }

    impl ValueReader for SynthReader {
        fn contigs(&self) -> &Genome {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            start: u64,
            end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            for range in self.ranges() {
                let lo = start.max(range.start);
                let hi = end.min(range.end);
                match &mut outputs[0] {
                    OutputSinkMut::I32(dst) => {
                        for pos in lo..hi {
                            dst[(pos - start) as usize] = (self.base + pos as i64) as i32;
                        }
                    }
                    OutputSinkMut::F32(dst) => {
                        for pos in lo..hi {
                            dst[(pos - start) as usize] = (self.base + pos as i64) as f32;
                        }
                    }
                    _ => panic!("test reader supports i32/f32 sinks only"),
                }
            }
            Ok(())
        }

        fn may_have_data(&self, _contig: &str, start: u64, end: u64) -> bool {
            self.ranges().any(|r| start < r.end && end > r.start)
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(self.clone())
        }
    }

    fn one_contig(len: u64) -> Genome {
        Genome::new(vec![Contig {
            name: "chr1".into(),
            length: len,
        }])
        .unwrap()
    }

    fn whole(store: &PbzStore, track: &str, len: u64) -> Region {
        Region {
            contig: store.genome_for(track).unwrap().id("chr1").unwrap(),
            start: 0,
            end: len,
        }
    }

    /// Number of chunk files (anything but `zarr.json`) under a values dir.
    fn chunk_file_count(values_dir: &Path) -> usize {
        fn walk(dir: &Path, n: &mut usize) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    walk(&entry.path(), n);
                } else if entry.file_name() != "zarr.json" {
                    *n += 1;
                }
            }
        }
        let mut n = 0;
        walk(values_dir, &mut n);
        n
    }

    #[test]
    fn auto_in_flight_targets_three_pieces_per_worker() {
        // ceil(3 * workers / readers), inside the clamp.
        assert_eq!(auto_in_flight_spans(64, 10), 20);
        assert_eq!(auto_in_flight_spans(4, 1), 12);
        assert_eq!(auto_in_flight_spans(22, 7), 10);
        // Clamp floor and ceiling.
        assert_eq!(auto_in_flight_spans(1, 64), 8);
        assert_eq!(auto_in_flight_spans(256, 1), 256);
    }

    #[test]
    fn auto_decode_chunks_rule() {
        // Cohort at 16k chunks: capped at 4M positions per span.
        assert_eq!(auto_decode_chunks(3_100_000_000, 16384, 100, 128), 256);
        // Small genome: balance rule, at least one chunk.
        assert_eq!(auto_decode_chunks(100, 16, 1, 4), 1);
        // 128 workers on one reader: balance rule below the cap.
        assert_eq!(auto_decode_chunks(3_100_000_000, 16384, 1, 128), 184);
        // Never more chunks than the genome has.
        assert_eq!(auto_decode_chunks(1000, 16, 64, 1), 63);
    }

    /// A sharded small-chunk import with multi-chunk decode spans reads back
    /// identical to the default-chunk import and to the unsharded one, and
    /// the piece count follows ceil(total / span) * readers.
    #[test]
    fn decode_spans_match_single_chunk_spans() {
        let len = 100u64;
        let genome = one_contig(len);
        let labels: Vec<String> = (0..4).map(|i| format!("s{i}")).collect();
        let readers = || -> Vec<SynthReader> {
            (0..4)
                .map(|i| SynthReader::new(genome.clone(), Dtype::I32, 5..95, (i + 1) * 1000))
                .collect()
        };
        let run = |dir: &Path, name: &str, cfg: TrackConfig, decode_chunks: usize| {
            let mut store = PbzStore::create(dir.join(name)).unwrap();
            store
                .create_track("v", genome.clone(), cfg.columns(labels.clone()))
                .unwrap();
            let report = Import::from_readers(readers())
                .unwrap()
                .into_track(store.track("v").unwrap())
                .readers_as_columns()
                .expect_column_labels(labels.clone())
                .options(PipelineOptions {
                    workers: 3,
                    decode_chunks,
                    ..PipelineOptions::default()
                })
                .run()
                .unwrap();
            (store, report)
        };
        let sharded = || {
            TrackConfig::new(Dtype::I32)
                .chunk_size(16)
                .column_chunk_size(2)
                .shard_size(48)
                .shard_column_size(2)
        };
        let dir = TempDir::new().unwrap();
        let (a, ra) = run(dir.path(), "a.pbz", sharded(), 3);
        let (b, _) = run(dir.path(), "b.pbz", sharded(), 1);
        let (c, _) = run(
            dir.path(),
            "c.pbz",
            TrackConfig::new(Dtype::I32)
                .chunk_size(16)
                .column_chunk_size(2),
            3,
        );
        assert_eq!(ra.pieces, (len.div_ceil(48) as usize) * 4);
        let read = |s: &PbzStore| -> ndarray::Array2<i32> {
            s.track("v")
                .unwrap()
                .read_region::<i32>(&whole(s, "v", len))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap()
        };
        assert_eq!(read(&a), read(&b));
        assert_eq!(read(&a), read(&c));
    }

    /// A chunk no reader covers still elides when it sits inside a covered
    /// decode span.
    #[test]
    fn uncovered_chunk_inside_covered_span_elides() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(64);
        let mut store = PbzStore::create(dir.path().join("elide.pbz")).unwrap();
        store
            .create_track(
                "v",
                genome.clone(),
                TrackConfig::new(Dtype::I32).chunk_size(16),
            )
            .unwrap();
        // Covered: chunk 0 (0..16) and chunk 3 (48..64); chunks 1 and 2 empty.
        let reader =
            SynthReader::new(genome.clone(), Dtype::I32, 0..16, 7).with_extra_cover(48..64);
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("v").unwrap())
            .options(PipelineOptions {
                workers: 1,
                decode_chunks: 4,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.pieces, 1);
        assert_eq!(report.tasks_skipped, 2);
        assert_eq!(chunk_file_count(&dir.path().join("elide.pbz/v/values")), 2);
    }

    /// A decode span crossing a contig boundary mid-chunk: the split chunk
    /// takes two windows from one piece, closes once, and keeps each contig's
    /// rows in place.
    #[test]
    fn span_crossing_contig_boundary_places_both_contigs() {
        let dir = TempDir::new().unwrap();
        let genome = Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 10,
            },
            Contig {
                name: "chr2".into(),
                length: 22,
            },
        ])
        .unwrap();
        let mut store = PbzStore::create(dir.path().join("split.pbz")).unwrap();
        store
            .create_track(
                "v",
                genome.clone(),
                TrackConfig::new(Dtype::I32).chunk_size(8),
            )
            .unwrap();

        let reader = SynthReader::new(genome, Dtype::I32, 0..22, 100);
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("v").unwrap())
            .options(PipelineOptions {
                workers: 1,
                decode_chunks: 4,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.pieces, 1);
        assert_eq!(report.tasks_completed, 4);
        assert_eq!(report.tasks_skipped, 0);

        for (contig, len) in [("chr1", 10u64), ("chr2", 22u64)] {
            let region = Region {
                contig: store.genome_for("v").unwrap().id(contig).unwrap(),
                start: 0,
                end: len,
            };
            let values = store
                .track("v")
                .unwrap()
                .read_region::<i32>(&region)
                .unwrap()
                .into_dimensionality::<Ix1>()
                .unwrap();
            for pos in 0..len as usize {
                assert_eq!(values[pos], 100 + pos as i32, "{contig} position {pos}");
            }
        }
    }

    #[test]
    fn nan_fill_prefill_and_elision() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(40);
        let mut store = PbzStore::create(dir.path().join("fill.pbz")).unwrap();
        // F32 tracks default to a NaN fill.
        store
            .create_track(
                "signal",
                genome.clone(),
                TrackConfig::new(Dtype::F32).chunk_size(16),
            )
            .unwrap();

        let reader = SynthReader::new(genome, Dtype::F32, 0..10, 0);
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("signal").unwrap())
            .run()
            .unwrap();
        // Chunks 16,16,8: only the first has coverage.
        assert_eq!(report.tasks_completed, 1);
        assert_eq!(report.tasks_skipped, 2);

        let values = store
            .track("signal")
            .unwrap()
            .read_region::<f32>(&whole(&store, "signal", 40))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        for pos in 0..10 {
            assert_eq!(values[pos], pos as f32);
        }
        // Uncovered positions of the WRITTEN buffer (10..16) and of the
        // ELIDED buffers (16..40) both read back as the NaN fill.
        for pos in 10..40 {
            assert!(values[pos].is_nan(), "position {pos} should be NaN");
        }
        assert_eq!(
            chunk_file_count(&dir.path().join("fill.pbz/signal/values")),
            1
        );
    }

    /// `ReverseCost` reverses decode span open order, so buffers complete out
    /// of genome order; the written data must be unaffected.
    #[test]
    fn out_of_order_buffer_completion() {
        struct ReverseCost;
        impl CostModel for ReverseCost {
            fn span_cost(&self, _contig: &str, start: u64, _end: u64) -> u64 {
                start
            }
        }

        let dir = TempDir::new().unwrap();
        let genome = one_contig(64);
        let mut store = PbzStore::create(dir.path().join("order.pbz")).unwrap();
        store
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32).chunk_size(16),
            )
            .unwrap();

        let reader = SynthReader::new(genome, Dtype::I32, 0..64, 100);
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("depth").unwrap())
            .with_cost_model(Arc::new(ReverseCost))
            .options(PipelineOptions {
                workers: 2,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.tasks_completed, 4);
        assert_eq!(report.tasks_skipped, 0);

        let values = store
            .track("depth")
            .unwrap()
            .read_region::<i32>(&whole(&store, "depth", 64))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        for pos in 0..64 {
            assert_eq!(values[pos], 100 + pos as i32);
        }
    }

    #[test]
    fn tap_reports_filled_and_skipped_buffers() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(48);
        let mut store = PbzStore::create(dir.path().join("tap.pbz")).unwrap();
        store
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32).chunk_size(16),
            )
            .unwrap();

        let (tap_tx, tap_rx) = bounded::<TapMessage>(16);
        let reader = SynthReader::new(genome, Dtype::I32, 16..32, 0);
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("depth").unwrap())
            .with_tap(tap_tx)
            .run()
            .unwrap();
        assert_eq!(report.tasks_completed, 1);
        assert_eq!(report.tasks_skipped, 2);

        let messages: Vec<TapMessage> = tap_rx.try_iter().collect();
        let filled: Vec<_> = messages
            .iter()
            .filter(|m| matches!(m, TapMessage::Filled { .. }))
            .collect();
        let skipped: Vec<_> = messages
            .iter()
            .filter(|m| matches!(m, TapMessage::Skipped { .. }))
            .collect();
        assert_eq!(filled.len(), 1);
        assert_eq!(skipped.len(), 2);
        assert_eq!(
            filled[0],
            &TapMessage::Filled {
                track: "depth".into(),
                position: 16..32,
                columns: 0..1,
            }
        );
    }

    /// Reader whose `read_into` records how many decodes run at once. Each
    /// call waits (bounded) for a peer, so serialized decodes stay at 1.
    #[derive(Clone)]
    struct ConcurrencyProbe {
        genome: Genome,
        schema: OutputSchema,
        /// `(currently decoding, max observed)`.
        gauge: Arc<(Mutex<(usize, usize)>, Condvar)>,
    }

    impl ValueReader for ConcurrencyProbe {
        fn contigs(&self) -> &Genome {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            _start: u64,
            _end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            let (lock, cv) = &*self.gauge;
            let mut g = lock.lock().unwrap();
            g.0 += 1;
            g.1 = g.1.max(g.0);
            cv.notify_all();
            let deadline = Instant::now() + Duration::from_secs(10);
            while g.1 < 2 {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                g = cv.wait_timeout(g, deadline - now).unwrap().0;
            }
            g.0 -= 1;
            cv.notify_all();
            drop(g);
            match &mut outputs[0] {
                OutputSinkMut::I32(dst) => dst.fill(1),
                _ => panic!("probe reader supports i32 sinks only"),
            }
            Ok(())
        }

        fn may_have_data(&self, _contig: &str, _start: u64, _end: u64) -> bool {
            true
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(self.clone())
        }
    }

    /// Two readers routed as columns of one column chunk share one buffer
    /// slot; their decodes must run concurrently, not serialized behind the
    /// slot lock.
    #[test]
    fn pieces_sharing_a_buffer_decode_concurrently() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(16);
        let labels = vec!["a".to_string(), "b".to_string()];
        let mut store = PbzStore::create(dir.path().join("conc.pbz")).unwrap();
        store
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(16)
                    .column_chunk_size(2),
            )
            .unwrap();

        let gauge = Arc::new((Mutex::new((0usize, 0usize)), Condvar::new()));
        let readers: Vec<ConcurrencyProbe> = (0..2)
            .map(|_| ConcurrencyProbe {
                genome: genome.clone(),
                schema: OutputSchema::single("value", Dtype::I32),
                gauge: Arc::clone(&gauge),
            })
            .collect();

        Import::from_readers(readers)
            .unwrap()
            .into_track(store.track("depth").unwrap())
            .readers_as_columns()
            .expect_column_labels(labels)
            .options(PipelineOptions {
                workers: 2,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();

        let max = gauge.0.lock().unwrap().1;
        assert!(
            max >= 2,
            "pieces sharing a buffer decoded serially: observed max concurrency {max}"
        );
    }

    /// A sharded cohort import must read back identical to an unsharded import
    /// of the same sources. The geometry exercises full interior shards,
    /// several subchunks per shard, a partial edge shard, and a column tail
    /// chunk boundary.
    #[test]
    fn sharded_cohort_matches_unsharded() {
        let len = 100u64;
        let genome = one_contig(len);
        let labels: Vec<String> = (0..4).map(|i| format!("s{i}")).collect();
        let readers = || -> Vec<SynthReader> {
            (0..4)
                .map(|i| SynthReader::new(genome.clone(), Dtype::I32, 5..95, (i + 1) * 1000))
                .collect()
        };

        let dir = TempDir::new().unwrap();
        let mut plain = PbzStore::create(dir.path().join("plain.pbz")).unwrap();
        plain
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(16)
                    .column_chunk_size(2),
            )
            .unwrap();
        let mut sharded = PbzStore::create(dir.path().join("sharded.pbz")).unwrap();
        sharded
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(16)
                    .column_chunk_size(2)
                    .shard_size(32)
                    .shard_column_size(4),
            )
            .unwrap();

        for store in [&plain, &sharded] {
            let report = Import::from_readers(readers())
                .unwrap()
                .into_track(store.track("depth").unwrap())
                .readers_as_columns()
                .expect_column_labels(labels.clone())
                .options(PipelineOptions {
                    workers: 4,
                    ..PipelineOptions::default()
                })
                .run()
                .unwrap();
            assert!(report.tasks_completed > 0);
        }

        let plain_data = plain
            .track("depth")
            .unwrap()
            .read_region::<i32>(&whole(&plain, "depth", len))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let sharded_data = sharded
            .track("depth")
            .unwrap()
            .read_region::<i32>(&whole(&sharded, "depth", len))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(plain_data, sharded_data);

        // Spot-check the source formula, and the zero fill outside the
        // covered range.
        for column in 0..4usize {
            let base = (column as i32 + 1) * 1000;
            assert_eq!(plain_data[[0, column]], 0);
            assert_eq!(plain_data[[5, column]], base + 5);
            assert_eq!(plain_data[[94, column]], base + 94);
            assert_eq!(plain_data[[95, column]], 0);
        }
        // Sharding must be the sole outer codec: that is what makes the
        // per-call `experimental_partial_encoding` append path available.
        assert_eq!(
            sharded.track("depth").unwrap().write_unit_shape().unwrap(),
            vec![32, 4]
        );
        assert!(
            sharded
                .track("depth")
                .unwrap()
                .values_array()
                .unwrap()
                .is_exclusively_sharded()
        );
    }

    /// Width-1 column chunks on a sharded track: the batched chunk flush must
    /// round-trip identical to the unsharded import, split correctly across
    /// column shards, and still elide fully uncovered chunks.
    #[test]
    fn sharded_width_one_columns_match_unsharded() {
        let len = 96u64;
        let genome = one_contig(len);
        let labels: Vec<String> = (0..4).map(|i| format!("s{i}")).collect();
        let readers = || -> Vec<SynthReader> {
            (0..4)
                .map(|i| SynthReader::new(genome.clone(), Dtype::I32, 5..60, (i + 1) * 1000))
                .collect()
        };

        let dir = TempDir::new().unwrap();
        let mut plain = PbzStore::create(dir.path().join("plain.pbz")).unwrap();
        plain
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(16)
                    .column_chunk_size(1),
            )
            .unwrap();
        let mut sharded = PbzStore::create(dir.path().join("sharded.pbz")).unwrap();
        sharded
            .create_track(
                "depth",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(16)
                    .column_chunk_size(1)
                    .shard_size(32)
                    .shard_column_size(2),
            )
            .unwrap();

        for store in [&plain, &sharded] {
            let report = Import::from_readers(readers())
                .unwrap()
                .into_track(store.track("depth").unwrap())
                .readers_as_columns()
                .expect_column_labels(labels.clone())
                .options(PipelineOptions {
                    workers: 4,
                    ..PipelineOptions::default()
                })
                .run()
                .unwrap();
            // 6 chunks x 4 column chunks; chunks 64..96 have no coverage.
            assert_eq!(report.tasks_completed, 16);
            assert_eq!(report.tasks_skipped, 8);
        }

        let plain_data = plain
            .track("depth")
            .unwrap()
            .read_region::<i32>(&whole(&plain, "depth", len))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let sharded_data = sharded
            .track("depth")
            .unwrap()
            .read_region::<i32>(&whole(&sharded, "depth", len))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(plain_data, sharded_data);

        // 3 x 2 shard grid; the uncovered position shard (64..96) elides, so
        // its two shard files are never created.
        assert_eq!(
            chunk_file_count(&dir.path().join("sharded.pbz/depth/values")),
            4
        );
    }

    /// Counts live forks of one reader so a test can assert the pool cap.
    struct ForkCounter {
        genome: Genome,
        schema: OutputSchema,
        live: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    impl Drop for ForkCounter {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl ValueReader for ForkCounter {
        fn contigs(&self) -> &Genome {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            _start: u64,
            _end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            thread::sleep(Duration::from_millis(5));
            outputs[0].as_i32_mut()?.fill(1);
            Ok(())
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            let n = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(n, Ordering::SeqCst);
            Ok(Self {
                genome: self.genome.clone(),
                schema: self.schema.clone(),
                live: Arc::clone(&self.live),
                peak: Arc::clone(&self.peak),
            })
        }
    }

    #[test]
    fn auto_handle_budget_reserves_margin() {
        assert_eq!(auto_handle_budget(1024, 10), (1024 * 3 / 4 - 128) / 2);
        // The floor is one handle per reader.
        assert_eq!(auto_handle_budget(256, 100), 100);
    }

    /// One reader, many workers, a budget of two handles: forks never exceed
    /// the pool size, every piece completes, and the wait is accounted.
    #[test]
    fn handle_pool_caps_forks_and_completes() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(64);
        let mut store = PbzStore::create(dir.path().join("pool.pbz")).unwrap();
        store
            .create_track(
                "v",
                genome.clone(),
                TrackConfig::new(Dtype::I32).chunk_size(4),
            )
            .unwrap();

        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let reader = ForkCounter {
            genome: genome.clone(),
            schema: OutputSchema::single("value", Dtype::I32),
            live: Arc::clone(&live),
            peak: Arc::clone(&peak),
        };
        let report = Import::from_readers(vec![reader])
            .unwrap()
            .into_track(store.track("v").unwrap())
            .options(PipelineOptions {
                workers: 8,
                handle_budget: 2,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 2, "peak forks {peak}");
        assert_eq!(report.tasks_completed, 16);
        // Six of the eight workers stall on the two handles, so the stall
        // outweighs the work. Counting the stall as busy would invert this.
        assert!(
            report.worker_busy_seconds < report.handle_wait_seconds,
            "handle wait leaked into worker busy time: busy {} vs wait {}",
            report.worker_busy_seconds,
            report.handle_wait_seconds
        );

        let values = store
            .track("v")
            .unwrap()
            .read_region::<i32>(&whole(&store, "v", 64))
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert!(values.iter().all(|&x| x == 1));
    }

    /// Budget below the reader count: every reader still gets one handle.
    #[test]
    fn handle_budget_floor_finishes_import() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(32);
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut store = PbzStore::create(dir.path().join("floor.pbz")).unwrap();
        store
            .create_track(
                "v",
                genome.clone(),
                TrackConfig::new(Dtype::I32)
                    .columns(labels.clone())
                    .chunk_size(8),
            )
            .unwrap();

        let readers: Vec<SynthReader> = (0..3)
            .map(|i| SynthReader::new(genome.clone(), Dtype::I32, 0..32, (i + 1) * 100))
            .collect();
        Import::from_readers(readers)
            .unwrap()
            .into_track(store.track("v").unwrap())
            .readers_as_columns()
            .expect_column_labels(labels)
            .options(PipelineOptions {
                workers: 4,
                handle_budget: 1,
                ..PipelineOptions::default()
            })
            .run()
            .unwrap();

        let values = store
            .track("v")
            .unwrap()
            .read_region::<i32>(&whole(&store, "v", 32))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(values[[5, 2]], 300 + 5);
    }
}
