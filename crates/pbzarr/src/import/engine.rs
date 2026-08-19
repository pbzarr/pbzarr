//! The import engine: one driver for every routing shape.
//!
//! A SPAN is one inner-chunk-length window of the flat position axis; a PIECE
//! is `(reader, span)`, the unit workers pull from the channel; a BUFFER is one
//! inner chunk of one target track (`(track, span, column chunk)`), pre-filled
//! with that track's fill value. The worker that finishes a buffer's last piece
//! writes it.
//!
//! Sharded tracks store one subchunk per buffer through
//! `store_array_subset_opt` with `experimental_partial_encoding` on: for a
//! subchunk-aligned subset the partial encoder reads only the shard index and
//! APPENDS the encoded subchunk instead of rewriting the shard. So each
//! subchunk must be written exactly once, and appends to one shard need a
//! per-shard mutex because every call rewrites that index.
//!
//! When no reader `may_have_data` over any of a buffer's pieces, the buffer is
//! never allocated and the write is elided: an absent chunk already reads back
//! as the fill value.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::Instant;

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
    /// Bounds peak scratch memory: open buffers never exceed this many spans'
    /// worth, plus the pieces individual workers are holding.
    pub in_flight_spans: usize,
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            workers: 4,
            in_flight_spans: 8,
            progress: None,
        }
    }
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
    /// `(schema field index, buffer-local column)`, strictly increasing by
    /// column.
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

struct EngineState {
    slots: Mutex<HashMap<SlotKey, Arc<SlotHandle>>>,
    gate: SpanGate,
    span_remaining: Vec<AtomicUsize>,
    bytes_written: AtomicU64,
    tasks_completed: AtomicUsize,
    tasks_skipped: AtomicUsize,
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

    /// Write or elide one closed buffer.
    fn close_buffer(
        &self,
        t_idx: usize,
        cc: u64,
        span: Range<u64>,
        data: Option<TileBuffer>,
        any_data: bool,
    ) -> Result<()> {
        let geom = &self.geoms[t_idx];
        let columns = geom.buffer_columns(cc);
        if !any_data {
            self.state.tasks_skipped.fetch_add(1, Ordering::Relaxed);
            if let Some(tap) = self.tap {
                // A dropped receiver detaches the tap; not an error.
                let _ = tap.send(TapMessage::Skipped {
                    track: geom.track.name().to_owned(),
                    position: span,
                    columns,
                });
            }
            return Ok(());
        }
        let data = data.ok_or_else(|| {
            PbzError::Metadata("engine invariant: covered buffer has no data".into())
        })?;

        if geom.sharded {
            let lock = self.shard_lock(t_idx, span.start, columns.start);
            let _guard = lock.lock().expect("shard mutex poisoned");
            data.store_subset(
                &geom.array,
                geom.track.name(),
                geom.rank,
                span.clone(),
                columns.clone(),
                &self.partial_opts,
            )?;
        } else {
            data.write_unsharded(geom.track, span.clone(), columns.clone())?;
        }

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
        Ok(())
    }
}

fn process_piece<R: ValueReader>(ctx: &RunCtx<'_>, reader: &mut R, piece: Piece) -> Result<()> {
    let windows = contig_windows(ctx.genome, piece.start, piece.end);
    let covered = windows
        .iter()
        .any(|w| reader.may_have_data(w.name, w.local_lo, w.local_hi));

    let entries = &ctx.touch_plan[piece.reader];
    let handles: Vec<Arc<SlotHandle>> = entries
        .iter()
        .map(|e| ctx.slot((e.t_idx, piece.span, e.cc)))
        .collect();
    let mut guards: Vec<MutexGuard<'_, SlotInner>> = handles
        .iter()
        .map(|h| h.inner.lock().expect("buffer slot poisoned"))
        .collect();

    if covered {
        let rows = (piece.end - piece.start) as usize;
        let n_fields = entries.iter().map(|e| e.fields.len()).sum::<usize>();
        for (guard, entry) in guards.iter_mut().zip(entries) {
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
        }
        for w in &windows {
            let mut staged: Vec<Option<OutputSinkMut<'_>>> = (0..n_fields).map(|_| None).collect();
            for (guard, entry) in guards.iter_mut().zip(entries) {
                let cols: Vec<usize> = entry.fields.iter().map(|&(_, c)| c).collect();
                let data = guard
                    .data
                    .as_mut()
                    .expect("buffer allocated above for covered piece");
                let sinks = data.column_sinks(w.row_lo..w.row_hi, &cols);
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
    }

    // Collect the buffers this piece finished, then write them outside the
    // slot locks.
    let mut closed: Vec<(usize, u64, Option<TileBuffer>, bool)> = Vec::new();
    for (guard, entry) in guards.iter_mut().zip(entries) {
        guard.remaining -= 1;
        guard.any_data |= covered;
        if guard.remaining == 0 {
            closed.push((entry.t_idx, entry.cc, guard.data.take(), guard.any_data));
        }
    }
    drop(guards);

    for (t_idx, cc, data, any_data) in closed {
        ctx.state
            .slots
            .lock()
            .expect("slot map poisoned")
            .remove(&(t_idx, piece.span, cc));
        ctx.close_buffer(t_idx, cc, piece.start..piece.end, data, any_data)?;
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

    let workers = options.workers.max(1);
    let started = Instant::now();
    let names: Vec<&str> = tracks.iter().map(|t| t.name()).collect();
    info!(
        "import pipeline: {} track(s) {names:?}, {total} positions, {n_spans} span(s) of \
         {chunk_len}, {n_readers} reader(s), {workers} workers",
        tracks.len()
    );

    let state = EngineState {
        slots: Mutex::new(HashMap::new()),
        gate: SpanGate::new(options.in_flight_spans),
        span_remaining: (0..n_spans).map(|_| AtomicUsize::new(n_readers)).collect(),
        bytes_written: AtomicU64::new(0),
        tasks_completed: AtomicUsize::new(0),
        tasks_skipped: AtomicUsize::new(0),
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
                while let Ok(piece) = piece_rx.recv() {
                    if ctx.state.has_err() {
                        // Drain so the channel closes cleanly; the gate is
                        // already aborted, so nothing blocks on us.
                        continue;
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
                                continue;
                            }
                        }
                    }
                    let reader = forks.get_mut(&piece.reader).expect("fork cached above");
                    if let Err(e) = process_piece(ctx, reader, piece) {
                        ctx.state.record_err(e);
                        continue;
                    }
                    if ctx.state.span_remaining[piece.span].fetch_sub(1, Ordering::AcqRel) == 1 {
                        ctx.state.gate.release();
                    }
                }
            });
        }

        'produce: for &span_idx in &span_order {
            if state.has_err() {
                break;
            }
            state.gate.acquire();
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

    if let Some(e) = state.first_err.lock().expect("error slot poisoned").take() {
        return Err(e);
    }

    Ok(Report {
        contigs_written: genome.iter().filter(|(_, c)| c.length > 0).count(),
        bytes_written: bytes,
        tasks_completed: completed,
        tasks_skipped: skipped,
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
}
