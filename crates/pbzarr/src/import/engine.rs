//! The import engine: one driver for every routing shape.
//!
//! A SPAN is one inner-chunk-length window of the flat position axis; a PIECE
//! is `(reader, span)`, the unit workers pull from the channel; a BUFFER is one
//! inner chunk of one target track (`(track, span, column chunk)`), pre-filled
//! with that track's fill value. The worker that finishes a buffer's last piece
//! writes it.
//!
//! Sharded tracks write through `store_array_subset_opt` with
//! `experimental_partial_encoding` on: for a subchunk-aligned subset the
//! partial encoder reads only the shard index, encodes the subchunks in
//! parallel, and APPENDS them instead of rewriting the shard. Each subchunk
//! must be written exactly once, and appends to one shard need a per-shard
//! mutex because every call rewrites that index. So a sharded span's buffers
//! flush together when its last buffer closes: one rectangular store call per
//! touched column shard.
//!
//! When no reader `may_have_data` over any of a buffer's pieces, the buffer is
//! never allocated. An unsharded buffer with no data elides its write; a
//! sharded span elides per column shard. An absent chunk already reads back
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
use crate::import::routing::{ImportRouting, SourceAxis, TrackTarget};
use crate::io::{Dtype, OutputSinkMut, ValueReader};
use crate::track::Track;

pub struct PipelineOptions {
    pub workers: usize,
    /// Open spans allowed at once. Bounds peak scratch memory: open buffers
    /// never exceed this many spans' worth, plus the pieces individual
    /// workers are holding. `0` = auto: `ceil(3 * workers / readers)`, so the
    /// runnable pieces target 3x the worker count, clamped to `[8, 256]` and
    /// to the span count.
    pub in_flight_spans: usize,
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            workers: 4,
            in_flight_spans: 0,
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

            /// Copies `src` (same dtype and row count, `cols.len()` columns
            /// wide) into the `cols` column range of this buffer.
            fn copy_columns_from(&mut self, src: &TileBuffer, cols: Range<usize>) -> Result<()> {
                match (self, src) {
                    $( (TileBuffer::$variant(dst), TileBuffer::$variant(src)) => {
                        dst.slice_mut(s![.., cols.start..cols.end]).assign(src);
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
    /// engine's span length.
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

/// One contig's overlap with a span.
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

type SlotKey = (usize, usize, u64); // (target, span, column chunk)

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

/// Closed buffers of one sharded `(target, span)`, held until the last one
/// arrives so the span flushes as one rectangle per column shard.
struct SpanGroup {
    /// `(column chunk, buffer)`; an uncovered buffer stays `None`.
    parts: Vec<(u64, Option<TileBuffer>)>,
    remaining: usize,
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
    /// Per piece: source read incl. contig windows and the coverage probe.
    decode: StageTimer,
    /// Per piece: the `may_have_data` probe alone (also inside `decode`).
    probe: StageTimer,
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
    /// Sharded `(target, span)` groups awaiting their last buffer.
    span_groups: Mutex<HashMap<(usize, usize), SpanGroup>>,
    /// Distinct column chunks per target: a sharded span group closes after
    /// this many buffers.
    group_sizes: Vec<usize>,
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

    fn record_skipped(&self, geom: &TargetGeom<'_>, span: Range<u64>, columns: Range<u64>) {
        self.state.tasks_skipped.fetch_add(1, Ordering::Relaxed);
        if let Some(tap) = self.tap {
            // A dropped receiver detaches the tap; not an error.
            let _ = tap.send(TapMessage::Skipped {
                track: geom.track.name().to_owned(),
                position: span,
                columns,
            });
        }
    }

    fn record_written(&self, geom: &TargetGeom<'_>, span: Range<u64>, columns: Range<u64>) {
        let bytes =
            (span.end - span.start) * (columns.end - columns.start) * dtype_bytes(geom.dtype);
        self.state.bytes_written.fetch_add(bytes, Ordering::Relaxed);
        self.state.tasks_completed.fetch_add(1, Ordering::Relaxed);
        if let Some(p) = self.progress {
            p.tick(bytes);
        }
        if let Some(tap) = self.tap {
            let _ = tap.send(TapMessage::Filled {
                track: geom.track.name().to_owned(),
                position: span,
                columns,
            });
        }
    }

    /// Write, batch, or elide one closed buffer.
    fn close_buffer(
        &self,
        t_idx: usize,
        cc: u64,
        span_idx: usize,
        span: Range<u64>,
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
            return self.close_sharded(t_idx, span_idx, span, cc, data);
        }
        let columns = geom.buffer_columns(cc);
        let Some(data) = data else {
            self.record_skipped(geom, span, columns);
            return Ok(());
        };
        let write_start = Instant::now();
        data.write_unsharded(geom.track, span.clone(), columns.clone())?;
        self.state.timings.write.record(write_start.elapsed());
        self.record_written(geom, span, columns);
        Ok(())
    }

    /// Stash one closed sharded buffer in its `(target, span)` group; the
    /// group flushes when its last buffer arrives. Every append into a shard
    /// rewrites the shard index under that shard's mutex, so the batch gives
    /// one index rewrite and one lock acquisition per span instead of one per
    /// buffer.
    fn close_sharded(
        &self,
        t_idx: usize,
        span_idx: usize,
        span: Range<u64>,
        cc: u64,
        data: Option<TileBuffer>,
    ) -> Result<()> {
        let done = {
            let mut groups = self.span_groups.lock().expect("span group map poisoned");
            let group = groups
                .entry((t_idx, span_idx))
                .or_insert_with(|| SpanGroup {
                    parts: Vec::with_capacity(self.group_sizes[t_idx]),
                    remaining: self.group_sizes[t_idx],
                });
            group.parts.push((cc, data));
            group.remaining -= 1;
            if group.remaining == 0 {
                groups.remove(&(t_idx, span_idx))
            } else {
                None
            }
        };
        match done {
            Some(group) => self.flush_span_group(t_idx, span, group),
            None => Ok(()),
        }
    }

    fn flush_span_group(&self, t_idx: usize, span: Range<u64>, group: SpanGroup) -> Result<()> {
        let geom = &self.geoms[t_idx];
        let rows = (span.end - span.start) as usize;
        let shard_w = geom.shard_col.max(1);
        let mut parts = group.parts;
        parts.sort_unstable_by_key(|&(cc, _)| cc);

        // A span is one subchunk long and subchunks tile the shard, so the
        // span rectangle can only straddle shards on the column axis.
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
                    self.record_skipped(geom, span.clone(), geom.buffer_columns(cc));
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
                        rect.copy_columns_from(
                            &data,
                            dst_lo..dst_lo + (cols.end - cols.start) as usize,
                        )?;
                    }
                }
                rect
            };

            let lock = self.shard_lock(t_idx, span.start, seg.start);
            let wait_start = Instant::now();
            let guard = lock.lock().expect("shard mutex poisoned");
            self.state.timings.shard_wait.record(wait_start.elapsed());
            let write_start = Instant::now();
            rect.store_subset(
                &geom.array,
                geom.track.name(),
                geom.rank,
                span.clone(),
                seg,
                &self.partial_opts,
            )?;
            self.state.timings.write.record(write_start.elapsed());
            drop(guard);

            for &(cc, _) in &parts[lo..hi] {
                self.record_written(geom, span.clone(), geom.buffer_columns(cc));
            }
            lo = hi;
        }
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
    let rows = (piece.end - piece.start) as usize;

    // Piece-local scratch, one buffer per touched entry, sized to this
    // piece's own columns. Peak scratch memory gains one such set per active
    // worker (typically a few MB each).
    let mut scratch: Vec<TileBuffer> = Vec::new();
    if covered {
        let read_start = Instant::now();
        let n_fields = entries.iter().map(|e| e.fields.len()).sum::<usize>();
        for entry in entries {
            let geom = &ctx.geoms[entry.t_idx];
            scratch.push(TileBuffer::filled(
                geom.dtype,
                rows,
                entry.fields.len(),
                &geom.fill,
            )?);
        }
        for w in &windows {
            let mut staged: Vec<Option<OutputSinkMut<'_>>> = (0..n_fields).map(|_| None).collect();
            for (local, entry) in scratch.iter_mut().zip(entries) {
                let cols: Vec<usize> = (0..entry.fields.len()).collect();
                let sinks = local.column_sinks(w.row_lo..w.row_hi, &cols);
                for (&(field, _), sink) in entry.fields.iter().zip(sinks) {
                    staged[field] = Some(sink);
                }
            }
            let mut outputs: Vec<OutputSinkMut<'_>> = staged
                .into_iter()
                .map(|s| s.expect("routing covers every schema field"))
                .collect();
            reader.read_into(w.name, w.local_lo, w.local_hi, &mut outputs)?;
        }
        decode += read_start.elapsed();
    }
    ctx.state.timings.decode.record(decode);

    // Merge into the shared buffers. Decode ran without slot locks so pieces
    // sharing a buffer stay concurrent; each lock covers only the region
    // copy. Buffers this piece finished are written outside the locks.
    let mut closed: Vec<(usize, u64, Option<TileBuffer>, bool)> = Vec::new();
    for (idx, entry) in entries.iter().enumerate() {
        let wait_start = Instant::now();
        let handle = ctx.slot((entry.t_idx, piece.span, entry.cc));
        let mut guard = handle.inner.lock().expect("buffer slot poisoned");
        ctx.state.timings.slot_wait.record(wait_start.elapsed());
        if covered {
            let merge_start = Instant::now();
            if guard.data.is_none() {
                let geom = &ctx.geoms[entry.t_idx];
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
                .expect("buffer allocated above for covered piece")
                .copy_columns_from(&scratch[idx], dst_lo..dst_lo + entry.fields.len())?;
            ctx.state.timings.merge.record(merge_start.elapsed());
        }
        guard.remaining -= 1;
        guard.any_data |= covered;
        if guard.remaining == 0 {
            closed.push((entry.t_idx, entry.cc, guard.data.take(), guard.any_data));
        }
    }

    for (t_idx, cc, data, any_data) in closed {
        ctx.state
            .slots
            .lock()
            .expect("slot map poisoned")
            .remove(&(t_idx, piece.span, cc));
        ctx.close_buffer(
            t_idx,
            cc,
            piece.span,
            piece.start..piece.end,
            data,
            any_data,
        )?;
    }
    Ok(())
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
    let chunk_len = geoms[0].array_chunk_len.max(1);
    let n_spans = usize::try_from(total.div_ceil(chunk_len))
        .map_err(|_| PbzError::Metadata("span count exceeds usize".into()))?;

    let spans: Vec<(u64, u64)> = (0..n_spans as u64)
        .map(|i| (i * chunk_len, ((i + 1) * chunk_len).min(total)))
        .collect();

    // Buffers and counters key on the span INDEX, so the open order is free
    // to change.
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

    let workers = options.workers.max(1);
    let in_flight = match options.in_flight_spans {
        0 => auto_in_flight_spans(workers, n_readers).min(n_spans.max(1)),
        n => n,
    };
    let started = Instant::now();
    let names: Vec<&str> = tracks.iter().map(|t| t.name()).collect();
    info!(
        "import pipeline: {} track(s) {names:?}, {total} positions, {n_spans} span(s) of \
         {chunk_len}, {n_readers} reader(s), {workers} workers, {in_flight} span(s) in flight",
        tracks.len()
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
        span_groups: Mutex::new(HashMap::new()),
        group_sizes,
    };

    // Workers fork lazily on first use and cache the fork by reader index, so
    // this lock is taken at most once per (worker, reader) pair.
    let originals = Mutex::new(readers);

    let (piece_tx, piece_rx) = bounded::<Piece>((workers * 2).max(1));

    thread::scope(|scope| {
        for _ in 0..workers {
            let piece_rx = piece_rx.clone();
            let ctx = &ctx;
            let originals = &originals;
            scope.spawn(move || {
                let mut forks: HashMap<usize, R> = HashMap::new();
                let mut busy = Duration::ZERO;
                let mut idle = Duration::ZERO;
                loop {
                    let wait_start = Instant::now();
                    let Ok(piece) = piece_rx.recv() else { break };
                    idle += wait_start.elapsed();
                    let work_start = Instant::now();
                    'piece: {
                        if ctx.state.has_err() {
                            // Drain so the channel closes cleanly; the gate is
                            // already aborted, so nothing blocks on us.
                            break 'piece;
                        }
                        if let std::collections::hash_map::Entry::Vacant(slot) =
                            forks.entry(piece.reader)
                        {
                            let fork = originals.lock().expect("readers poisoned")[piece.reader]
                                .fork()
                                .map_err(PbzError::Reader);
                            match fork {
                                Ok(fork) => {
                                    slot.insert(fork);
                                }
                                Err(e) => {
                                    ctx.state.record_err(e);
                                    break 'piece;
                                }
                            }
                        }
                        let reader = forks.get_mut(&piece.reader).expect("fork cached above");
                        if let Err(e) = process_piece(ctx, reader, piece) {
                            ctx.state.record_err(e);
                            break 'piece;
                        }
                        if ctx.state.span_remaining[piece.span].fetch_sub(1, Ordering::AcqRel) == 1
                        {
                            ctx.state.gate.release();
                        }
                    }
                    busy += work_start.elapsed();
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
         decode : total {:.1}s  pieces {}  mean {:.2}s  max {:.2}s\n\
         probe  : total {:.1}s\n\
         slot wait: total {:.1}s  waits {}  mean {:.2}s  max {:.2}s\n\
         merge  : total {:.1}s  copies {}  mean {:.2}s  max {:.2}s\n\
         shard wait: total {:.1}s  waits {}  mean {:.2}s  max {:.2}s\n\
         write  : total {:.1}s  stores {}  mean {:.2}s  max {:.2}s\n\
         worker busy {busy:.0}s / idle {idle:.0}s ({busy_pct:.0}% busy)\n\
         gate wait: {gate_wait:.1}s",
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
    );

    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    Ok(Report {
        contigs_written: genome.iter().filter(|(_, c)| c.length > 0).count(),
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
    /// `covered` reads as `base + p`; everything else is left untouched.
    #[derive(Clone)]
    struct SynthReader {
        genome: Genome,
        schema: OutputSchema,
        covered: Range<u64>,
        base: i64,
    }

    impl SynthReader {
        fn new(genome: Genome, dtype: Dtype, covered: Range<u64>, base: i64) -> Self {
            Self {
                genome,
                schema: OutputSchema::single("value", dtype),
                covered,
                base,
            }
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
            let lo = start.max(self.covered.start);
            let hi = end.min(self.covered.end);
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
            Ok(())
        }

        fn may_have_data(&self, _contig: &str, start: u64, end: u64) -> bool {
            start < self.covered.end && end > self.covered.start
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
        // Spans 16,16,8: only the first has coverage.
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

    /// `ReverseCost` reverses span open order, so buffers complete out of
    /// genome order; the written data must be unaffected.
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

    /// Width-1 column chunks on a sharded track: the batched span flush must
    /// round-trip identical to the unsharded import, split correctly across
    /// column shards, and still elide fully uncovered spans.
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
            // 6 spans x 4 column chunks; spans 64..96 have no coverage.
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
}
