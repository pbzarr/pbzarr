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
//! A sharded target whose destination columns cover whole shards encodes
//! each buffer as one inner chunk when it closes and keeps only the encoded
//! bytes in its shard's group. The group's last close lays out the shard and
//! its index and stores it with one call, so each shard file is written
//! exactly once and needs no lock. A sharded target whose
//! columns do not cover whole shards falls back to per-chunk rectangles
//! through the partial encoder, which appends subchunks under a per-shard
//! mutex because every append rewrites the shard.
//!
//! When no reader reports coverage for any of a buffer's pieces, the buffer
//! is never allocated. An unsharded buffer with no data elides its write; a whole
//! shard with no data elides its file; the fallback path elides per column
//! shard. An absent chunk already reads back as the fill value.

use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::num::NonZeroU64;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use log::{debug, info, warn};
use ndarray::{Array2, ArrayViewMut1, ArrayViewMut2, Axis, s};
use zarrs::array::codec::{ShardingCodecConfiguration, ShardingIndexLocation};
use zarrs::array::data_type::uint64;
use zarrs::array::{
    Array, ArrayBytes, ArrayMetadata, ArrayShardedExt, ArraySubset, ArrayToBytesCodecTraits,
    BytesRepresentation, CodecChain, CodecOptions, DataType, Element, FillValue,
};
use zarrs::storage::ReadableWritableListableStorageTraits;

use super::destination::Destination;
use crate::Result;
use crate::dataset::Reference;
use crate::error::PbzError;
use crate::import::config::{Config, ProgressSink, Report};
use crate::import::progress::human_bytes;
use crate::import::routing::ImportRouting;
use crate::io::{Dtype, OutputSinkMut, ReaderError, ValueReader, WindowSink};

/// Open decode spans allowed at once, before the span count cap the engine
/// applies. This bounds the open chunk buffers: each span holds up to
/// `decode_chunks` buffers per track. The runnable pieces target 3x the
/// worker count.
fn auto_in_flight_spans(workers: usize, n_readers: usize) -> usize {
    (3 * workers.max(1))
        .div_ceil(n_readers.max(1))
        .clamp(8, 256)
}

/// Decode span cap in positions: 4M keeps a piece near one second of
/// decode on a deep BAM while bounding open chunk buffers per span.
const DECODE_SPAN_CAP: u64 = 4 << 20;

/// Auto rule for `Config::decode_chunks` = 0: as many chunks as the
/// 4M cap allows, reduced so every worker sees about eight pieces.
fn auto_decode_chunks(total: u64, chunk_len: u64, n_readers: usize, workers: usize) -> usize {
    let chunk_len = chunk_len.max(1);
    let n_chunks = total.div_ceil(chunk_len).max(1);
    let cap = (DECODE_SPAN_CAP / chunk_len).max(1);
    let balance = total * n_readers.max(1) as u64 / (8 * workers.max(1) as u64 * chunk_len);
    usize::try_from(balance.clamp(1, cap).min(n_chunks)).unwrap_or(usize::MAX)
}

/// Snap spans to the shard grid of shard-aligned targets: a whole number of
/// shard rows per span, or a whole number of spans per shard row. A shard
/// group then finishes within the spans that opened it, so parked shard
/// buffers stay under the span gate's bound.
fn snap_decode_chunks(decode_chunks: usize, geoms: &[TargetGeom<'_>]) -> usize {
    let mut all_gcd: u64 = 0;
    let mut all_lcm: u64 = 1;
    for geom in geoms.iter().filter(|g| g.shard_aligned) {
        let cps = geom.chunks_per_shard.max(1);
        all_gcd = gcd(all_gcd, cps);
        all_lcm = (all_lcm / gcd(all_lcm, cps)).saturating_mul(cps);
    }
    if all_gcd == 0 {
        return decode_chunks;
    }
    let dc = decode_chunks as u64;
    let snapped = if dc >= all_lcm {
        dc - dc % all_lcm
    } else {
        (1..=dc)
            .rev()
            .find(|d| all_gcd.is_multiple_of(*d))
            .unwrap_or(1)
    };
    snapped.max(1) as usize
}

fn gcd(a: u64, b: u64) -> u64 {
    if a == 0 { b } else { gcd(b % a, a) }
}

/// Reader handles allowed across all readers at once, base readers included.
/// Two fds per handle (data + index); a quarter of the limit plus 128 fds
/// stay free for the store and shard files. Each reader's pool gets
/// `budget / readers`, at least 1, at most `workers`.
fn auto_handle_budget(soft_fd_limit: u64, n_readers: usize) -> usize {
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

fill_decode_num!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl FillDecode for bool {
    fn decode(fill: &FillValue) -> Result<Self> {
        Ok(fill.as_ne_bytes().first().copied().unwrap_or(0) != 0)
    }
}

/// One rank-1 view per column, borrowing the original lifetime.
fn split_columns<'a, T>(mut view: ArrayViewMut2<'a, T>) -> Vec<ArrayViewMut1<'a, T>> {
    let ncols = view.ncols();
    let mut out = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let (column, rest) = view.split_at(Axis(1), 1);
        out.push(column.remove_axis(Axis(1)));
        view = rest;
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

            /// One sink per buffer column, in column order.
            fn column_sinks(&mut self, rows: Range<usize>) -> Vec<OutputSinkMut<'_>> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        split_columns(a.slice_mut(s![rows.clone(), ..]))
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
            fn write_unsharded(self, track: &Destination, pos: Range<u64>, cols: Range<u64>) -> Result<()> {
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

            /// Encodes the buffer as one inner chunk of a shard. A clipped
            /// tail pads with fill. `None` when every value is the fill.
            fn encode_inner(&self, codec: &ShardCodec, fill: &FillValue) -> Result<Option<Vec<u8>>> {
                match self {
                    $( TileBuffer::$variant(a) => {
                        let shape = (codec.chunk_rows, codec.chunk_cols);
                        let padded;
                        let full = if a.dim() == shape && a.is_standard_layout() {
                            a
                        } else {
                            let mut p = Array2::from_elem(shape, <$ty as FillDecode>::decode(fill)?);
                            p.slice_mut(s![..a.nrows(), ..a.ncols()]).assign(a);
                            padded = p;
                            &padded
                        };
                        let values = full.as_slice().expect("standard layout");
                        let bytes = <$ty as Element>::to_array_bytes(&codec.data_type, values)
                            .map_err(|e| PbzError::Store(format!("encode inner chunk: {e}")))?;
                        codec.encode(bytes, fill)
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
    U64 => u64,
    I8 => i8,
    I16 => i16,
    I32 => i32,
    I64 => i64,
    F32 => f32,
    F64 => f64,
    Bool => bool,
}

/// The inner and index codecs of a shard-aligned target, so a worker can
/// encode each inner chunk as its buffer closes and write the shard as the
/// sharding codec lays it out.
struct ShardCodec {
    inner: CodecChain,
    index: CodecChain,
    index_location: ShardingIndexLocation,
    index_len: usize,
    data_type: DataType,
    /// `[rows]` or `[rows, columns]` of one inner chunk.
    inner_shape: Vec<NonZeroU64>,
    /// `[chunks along positions, (chunks along columns,) 2]`.
    index_shape: Vec<NonZeroU64>,
    chunk_rows: usize,
    chunk_cols: usize,
    /// Inner chunks per shard along the column axis; 1 for rank-1 tracks.
    col_chunks: u64,
    opts: CodecOptions,
}

impl ShardCodec {
    fn new(
        array: &Array<dyn ReadableWritableListableStorageTraits>,
        rank: usize,
        chunk_len: u64,
        col_chunk: u64,
        shard_pos: u64,
        shard_col: u64,
    ) -> Result<Self> {
        let err = |e: String| PbzError::Metadata(format!("sharding codec: {e}"));
        let ArrayMetadata::V3(meta) = array.metadata() else {
            return Err(err("expected zarr v3 metadata".into()));
        };
        let sharding = meta
            .codecs
            .iter()
            .find(|c| c.name() == "sharding_indexed")
            .ok_or_else(|| err("no sharding_indexed codec".into()))?;
        let ShardingCodecConfiguration::V1(config) = sharding
            .to_typed_configuration::<ShardingCodecConfiguration>()
            .map_err(|e| err(e.to_string()))?
        else {
            return Err(err("unsupported configuration version".into()));
        };
        let inner = CodecChain::from_metadata(&config.codecs).map_err(|e| err(e.to_string()))?;
        let index =
            CodecChain::from_metadata(&config.index_codecs).map_err(|e| err(e.to_string()))?;
        let nz = |v: u64| NonZeroU64::new(v).ok_or_else(|| err("zero chunk extent".into()));
        let col_chunks = if rank == 1 { 1 } else { shard_col / col_chunk };
        let (inner_shape, index_shape) = if rank == 1 {
            (
                vec![nz(chunk_len)?],
                vec![nz(shard_pos / chunk_len)?, nz(2)?],
            )
        } else {
            (
                vec![nz(chunk_len)?, nz(col_chunk)?],
                vec![nz(shard_pos / chunk_len)?, nz(col_chunks)?, nz(2)?],
            )
        };
        let index_len = match index
            .encoded_representation(&index_shape, &uint64(), &FillValue::from(u64::MAX))
            .map_err(|e| err(e.to_string()))?
        {
            BytesRepresentation::FixedSize(n) => n as usize,
            other => return Err(err(format!("shard index is not fixed size: {other}"))),
        };
        Ok(Self {
            inner,
            index,
            index_location: config.index_location,
            index_len,
            data_type: array.data_type().clone(),
            inner_shape,
            index_shape,
            chunk_rows: chunk_len as usize,
            chunk_cols: if rank == 1 { 1 } else { col_chunk as usize },
            col_chunks,
            opts: CodecOptions::default().with_concurrent_target(1),
        })
    }

    fn encode(&self, bytes: ArrayBytes<'_>, fill: &FillValue) -> Result<Option<Vec<u8>>> {
        if bytes.is_fill_value(fill) {
            return Ok(None);
        }
        let encoded = self
            .inner
            .encode(bytes, &self.inner_shape, &self.data_type, fill, &self.opts)
            .map_err(|e| PbzError::Store(format!("encode inner chunk: {e}")))?;
        // The compressor allocates for the uncompressed size; the shard
        // group keeps this until the shard is stored.
        let mut encoded = encoded.into_owned();
        encoded.shrink_to_fit();
        Ok(Some(encoded))
    }

    /// Lays out encoded inner chunks in index order plus the shard index,
    /// the same bytes the sharding codec writes with C write order.
    fn assemble(&self, mut chunks: Vec<(usize, Vec<u8>)>) -> Result<Vec<u8>> {
        chunks.sort_unstable_by_key(|(i, _)| *i);
        let n_index: usize = self.index_shape.iter().map(|d| d.get() as usize).product();
        let mut index = vec![u64::MAX; n_index];
        let data_len: usize = chunks.iter().map(|(_, c)| c.len()).sum();
        let mut out = Vec::with_capacity(data_len + self.index_len);
        let base = match self.index_location {
            ShardingIndexLocation::Start => self.index_len,
            ShardingIndexLocation::End => 0,
        };
        if base > 0 {
            out.resize(base, 0);
        }
        for (i, chunk) in &chunks {
            index[2 * i] = out.len() as u64;
            index[2 * i + 1] = chunk.len() as u64;
            out.extend_from_slice(chunk);
        }
        drop(chunks);
        let raw: Vec<u8> = index.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let encoded = self
            .index
            .encode(
                ArrayBytes::from(raw),
                &self.index_shape,
                &uint64(),
                &FillValue::from(u64::MAX),
                &self.opts,
            )
            .map_err(|e| PbzError::Store(format!("encode shard index: {e}")))?;
        if encoded.len() != self.index_len {
            return Err(PbzError::Store("shard index size mismatch".into()));
        }
        match self.index_location {
            ShardingIndexLocation::Start => out[..self.index_len].copy_from_slice(&encoded),
            ShardingIndexLocation::End => out.extend_from_slice(&encoded),
        }
        Ok(out)
    }
}

/// Resolved write geometry for one target track.
struct TargetGeom<'t> {
    track: &'t Destination,
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
    /// Sharding is the only codec and the destination columns cover whole
    /// shards, so every buffer of a shard arrives through this import and
    /// the engine can lay out the shard file itself.
    shard_aligned: bool,
    /// Inner position chunks per shard; meaningful only when `shard_aligned`.
    chunks_per_shard: u64,
    /// Set when `shard_aligned`.
    shard_codec: Option<ShardCodec>,
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

    /// Bytes one closed buffer adds to the progress total: its rows times
    /// its columns inside the target range, so a run's ticks sum to the
    /// `set_total` figure whether the buffer is written or skipped.
    fn progress_bytes(&self, chunk: &Range<u64>, cc: u64) -> u64 {
        let cols = self.buffer_columns(cc);
        let width = cols
            .end
            .min(self.columns.end)
            .saturating_sub(cols.start.max(self.columns.start));
        (chunk.end - chunk.start) * width * self.dtype.size() as u64
    }

    fn col_shard(&self, cc: u64) -> u64 {
        cc / (self.shard_col / self.col_chunk.max(1)).max(1)
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

fn contig_windows(reference: &Reference, start: u64, end: u64) -> Vec<ContigWindow<'_>> {
    let offsets = reference.offsets();
    let first = offsets.partition_point(|&o| o <= start).saturating_sub(1);
    let mut windows = Vec::new();
    for (i, contig) in reference.contigs().iter().enumerate().skip(first) {
        let c_start = offsets[i];
        if c_start >= end {
            break;
        }
        let c_end = offsets[i + 1];
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

    fn lock(&self) -> MutexGuard<'_, (usize, bool)> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn acquire(&self) {
        let mut s = self.lock();
        while s.0 >= self.cap && !s.1 {
            s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
        if !s.1 {
            s.0 += 1;
        }
    }

    fn release(&self) {
        let mut s = self.lock();
        s.0 = s.0.saturating_sub(1);
        drop(s);
        self.cv.notify_one();
    }

    fn abort(&self) {
        self.lock().1 = true;
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

/// The buffers each reader's pieces touch, and every count derived from them.
struct TouchPlan {
    /// Per reader index: the buffers a piece for that reader touches.
    entries: Vec<Vec<TouchEntry>>,
    /// `[target][column chunk]`: the pieces that feed one buffer.
    slot_pieces: Vec<Vec<usize>>,
    /// Distinct column chunks per target: a sharded chunk group closes after
    /// this many buffers.
    group_sizes: Vec<usize>,
    /// Per target: touched column chunks per column shard, for shard-aligned
    /// targets. Sizes the shard groups.
    col_shard_ccs: Vec<HashMap<u64, usize>>,
}

type SlotKey = (usize, usize, u64); // (target, chunk, column chunk)

/// `(target, position shard, column shard)` -> its append-serializing mutex.
type ShardLockMap = HashMap<(usize, u64, u64), Arc<Mutex<()>>>;

/// `(target, position shard, column shard)` -> its in-progress shard group.
type ShardGroupMap = HashMap<(usize, u64, u64), Arc<Mutex<ShardGroup>>>;

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

/// One shard of a shard-aligned target. Each closed buffer adds its encoded
/// inner chunk; the last one assembles and stores the shard.
struct ShardGroup {
    /// `(inner chunk index in the shard, encoded bytes)`; an all-fill chunk
    /// adds nothing.
    encoded: Vec<(usize, Vec<u8>)>,
    /// `(chunk index, column chunk, covered)`, recorded at flush.
    parts: Vec<(usize, u64, bool)>,
    remaining: usize,
}

/// Idle handles of one reader, capped at `capacity` live handles. The pool
/// starts with the base reader, so the base counts against the cap. A worker
/// that takes the last idle handle forks one more from it while the cap
/// allows, so a reader no worker ever decodes opens nothing new.
struct HandlePool<R> {
    /// `(idle handles, handles created, aborted)`.
    state: Mutex<(Vec<R>, usize, bool)>,
    cv: Condvar,
    capacity: usize,
}

impl<R: ValueReader> HandlePool<R> {
    fn new(base: R, capacity: usize) -> Self {
        Self {
            state: Mutex::new((vec![base], 1, false)),
            cv: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    fn lock(&self) -> MutexGuard<'_, (Vec<R>, usize, bool)> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Blocks while every handle is checked out; returns `Ok(None)` once the
    /// pool is aborted so a waiting worker can drain.
    fn checkout(&self) -> Result<Option<R>> {
        let mut guard = self.lock();
        loop {
            if let Some(handle) = guard.0.pop() {
                if !guard.0.is_empty() || guard.1 >= self.capacity {
                    return Ok(Some(handle));
                }
                guard.1 += 1;
                drop(guard);
                let forked = handle.fork();
                let mut guard = self.lock();
                return match forked {
                    Ok(fork) => {
                        guard.0.push(fork);
                        drop(guard);
                        self.cv.notify_one();
                        Ok(Some(handle))
                    }
                    Err(e) => {
                        guard.1 -= 1;
                        guard.0.push(handle);
                        drop(guard);
                        self.cv.notify_one();
                        Err(PbzError::Reader(e))
                    }
                };
            }
            if guard.2 {
                return Ok(None);
            }
            guard = self.cv.wait(guard).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn checkin(&self, handle: R) {
        self.lock().0.push(handle);
        self.cv.notify_one();
    }

    fn abort(&self) {
        self.lock().2 = true;
        self.cv.notify_all();
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

    fn summary(&self, noun: &str) -> String {
        format!(
            "total {:.1}s  {noun} {}  mean {:.2}s  max {:.2}s",
            self.total_secs(),
            self.count(),
            self.mean_secs(),
            self.max_secs()
        )
    }
}

#[derive(Default)]
struct RunTimings {
    /// Per piece: the source read, and the merges and chunk closes the
    /// stream runs inside it.
    decode: StageTimer,
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
    /// Per closed shard-aligned buffer: its inner chunk encode.
    encode: StageTimer,
    /// Per worker thread: its whole time on pieces, pool stalls excluded.
    worker_busy: StageTimer,
    /// Per worker thread: its whole time waiting for a piece.
    worker_idle: StageTimer,
}

impl RunTimings {
    fn log(&self, wall: Duration, workers: usize, gate_wait: Duration) {
        let wall = wall.as_secs_f64();
        let busy = self.worker_busy.total_secs();
        let idle = self.worker_idle.total_secs();
        let gate_wait = gate_wait.as_secs_f64();
        let busy_pct = if busy + idle > 0.0 {
            100.0 * busy / (busy + idle)
        } else {
            0.0
        };
        info!(
            "import timing: wall {wall:.1}s workers {workers}\n\
             decode (incl. merge+write): {}\n\
             slot wait: {}\n\
             merge  : {}\n\
             shard wait: {}\n\
             write  : {}\n\
             encode : {}\n\
             worker busy {busy:.0}s / idle {idle:.0}s ({busy_pct:.0}% busy)\n\
             gate wait: {gate_wait:.1}s\n\
             handle wait: {}",
            self.decode.summary("pieces"),
            self.slot_wait.summary("waits"),
            self.merge.summary("copies"),
            self.shard_wait.summary("waits"),
            self.write.summary("stores"),
            self.encode.summary("chunks"),
            self.handle_wait.summary("waits"),
        );
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload")
}

/// Everything workers share.
struct RunCtx<'a> {
    slots: Mutex<HashMap<SlotKey, Arc<SlotHandle>>>,
    gate: SpanGate,
    span_remaining: Vec<AtomicUsize>,
    bytes_written: AtomicU64,
    chunks_written: AtomicUsize,
    chunks_skipped: AtomicUsize,
    timings: RunTimings,
    err_flag: AtomicBool,
    first_err: Mutex<Option<PbzError>>,
    geoms: &'a [TargetGeom<'a>],
    reference: &'a Reference,
    plan: &'a TouchPlan,
    progress: Option<&'a dyn ProgressSink>,
    cancel: Option<&'a AtomicBool>,
    partial_opts: CodecOptions,
    shard_locks: Mutex<ShardLockMap>,
    /// Sharded `(target, chunk)` groups awaiting their last buffer (the
    /// fallback path for targets that are not shard-aligned).
    chunk_groups: Mutex<HashMap<(usize, usize), ChunkGroup>>,
    /// Shard groups awaiting their last buffer, for shard-aligned targets.
    /// Each group carries its own lock so closers merge concurrently across
    /// shards.
    shard_groups: Mutex<ShardGroupMap>,
    /// Flat position count, so a piece can clamp its last chunk.
    total: u64,
}

impl RunCtx<'_> {
    fn has_err(&self) -> bool {
        self.err_flag.load(Ordering::Relaxed)
    }

    /// Record `PbzError::Cancelled` when the caller set the cancel flag.
    fn check_cancel(&self) -> bool {
        let cancelled = self.cancel.is_some_and(|flag| flag.load(Ordering::Relaxed));
        if cancelled {
            self.record_err(PbzError::Cancelled);
        }
        cancelled
    }

    fn record_err(&self, err: PbzError) {
        let mut slot = self
            .first_err
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
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

    /// Every target shares one inner position chunk length.
    fn chunk_len(&self) -> u64 {
        self.geoms[0].array_chunk_len
    }

    /// Flat positions of chunk `idx`, clamped at the end of the genome.
    fn chunk_range(&self, idx: u64) -> Range<u64> {
        let lo = idx * self.chunk_len();
        lo..(lo + self.chunk_len()).min(self.total)
    }

    fn slot(&self, key: SlotKey) -> Arc<SlotHandle> {
        let mut map = self.slots.lock().expect("slot map poisoned");
        Arc::clone(map.entry(key).or_insert_with(|| {
            Arc::new(SlotHandle {
                inner: Mutex::new(SlotInner {
                    data: None,
                    remaining: self.plan.slot_pieces[key.0][key.2 as usize],
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

    /// Count one closed buffer as written or elided.
    fn record_close(
        &self,
        geom: &TargetGeom<'_>,
        chunk: Range<u64>,
        columns: Range<u64>,
        written: bool,
    ) {
        if written {
            let bytes = (chunk.end - chunk.start)
                * (columns.end - columns.start)
                * geom.dtype.size() as u64;
            self.bytes_written.fetch_add(bytes, Ordering::Relaxed);
            self.chunks_written.fetch_add(1, Ordering::Relaxed);
        } else {
            self.chunks_skipped.fetch_add(1, Ordering::Relaxed);
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
        let progress_bytes = geom.progress_bytes(&chunk, cc);
        match (geom.sharded, geom.shard_aligned) {
            (false, _) => {
                let columns = geom.buffer_columns(cc);
                let written = data.is_some();
                if let Some(data) = data {
                    let write_start = Instant::now();
                    data.write_unsharded(geom.track, chunk.clone(), columns.clone())?;
                    self.timings.write.record(write_start.elapsed());
                }
                self.record_close(geom, chunk, columns, written);
            }
            (true, true) => self.close_shard_part(t_idx, chunk_idx, cc, data)?,
            (true, false) => self.park_in_chunk_group(t_idx, chunk_idx, chunk, cc, data)?,
        }
        if let Some(p) = self.progress {
            p.tick(progress_bytes);
        }
        Ok(())
    }

    /// Stash one closed sharded buffer in its `(target, chunk)` group; the
    /// group flushes when its last buffer arrives. Every append into a shard
    /// rewrites the shard index under that shard's mutex, so the batch gives
    /// one index rewrite and one lock acquisition per chunk instead of one per
    /// buffer.
    fn park_in_chunk_group(
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
                    parts: Vec::with_capacity(self.plan.group_sizes[t_idx]),
                    remaining: self.plan.group_sizes[t_idx],
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
                    self.record_close(geom, chunk.clone(), geom.buffer_columns(cc), false);
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
            self.timings.shard_wait.record(wait_start.elapsed());
            let write_start = Instant::now();
            rect.store_subset(
                &geom.array,
                geom.track.name(),
                geom.rank,
                chunk.clone(),
                seg,
                &self.partial_opts,
            )?;
            self.timings.write.record(write_start.elapsed());
            drop(guard);

            for &(cc, _) in &parts[lo..hi] {
                self.record_close(geom, chunk.clone(), geom.buffer_columns(cc), true);
            }
            lo = hi;
        }
        Ok(())
    }

    /// Encode one closed buffer into its shard's group; the last part stores
    /// the shard.
    fn close_shard_part(
        &self,
        t_idx: usize,
        chunk_idx: usize,
        cc: u64,
        data: Option<TileBuffer>,
    ) -> Result<()> {
        let geom = &self.geoms[t_idx];
        let codec = geom
            .shard_codec
            .as_ref()
            .expect("shard-aligned target has a shard codec");
        let cps = geom.chunks_per_shard.max(1);
        let row = chunk_idx as u64 / cps;
        let cs = geom.col_shard(cc);
        let key = (t_idx, row, cs);
        let covered = data.is_some();
        let encoded = match data {
            Some(buffer) => {
                let encode_start = Instant::now();
                let encoded = buffer.encode_inner(codec, &geom.fill)?;
                self.timings.encode.record(encode_start.elapsed());
                encoded
            }
            None => None,
        };
        let inner_idx = ((chunk_idx as u64 - row * cps) * codec.col_chunks
            + (cc - cs * codec.col_chunks)) as usize;
        let handle = {
            let mut groups = self.shard_groups.lock().expect("shard group map poisoned");
            match groups.entry(key) {
                Entry::Occupied(e) => Arc::clone(e.get()),
                Entry::Vacant(e) => {
                    let n_chunks = self.total.div_ceil(self.chunk_len());
                    let row_chunks = (n_chunks.min((row + 1) * cps) - row * cps) as usize;
                    let ccs = *self.plan.col_shard_ccs[t_idx].get(&cs).ok_or_else(|| {
                        PbzError::Metadata(
                            "engine invariant: closed buffer outside the touch plan".into(),
                        )
                    })?;
                    let expected = row_chunks * ccs;
                    Arc::clone(e.insert(Arc::new(Mutex::new(ShardGroup {
                        encoded: Vec::new(),
                        parts: Vec::with_capacity(expected),
                        remaining: expected,
                    }))))
                }
            }
        };
        let done = {
            let mut group = handle.lock().expect("shard group poisoned");
            if let Some(bytes) = encoded {
                group.encoded.push((inner_idx, bytes));
            }
            group.parts.push((chunk_idx, cc, covered));
            group.remaining -= 1;
            (group.remaining == 0).then(|| {
                (
                    std::mem::take(&mut group.encoded),
                    std::mem::take(&mut group.parts),
                )
            })
        };
        if let Some((encoded, parts)) = done {
            self.shard_groups
                .lock()
                .expect("shard group map poisoned")
                .remove(&key);
            self.flush_shard(t_idx, row, cs, encoded, parts)?;
        }
        Ok(())
    }

    fn flush_shard(
        &self,
        t_idx: usize,
        row: u64,
        cs: u64,
        encoded: Vec<(usize, Vec<u8>)>,
        parts: Vec<(usize, u64, bool)>,
    ) -> Result<()> {
        let geom = &self.geoms[t_idx];
        let codec = geom
            .shard_codec
            .as_ref()
            .expect("shard-aligned target has a shard codec");
        let indices: &[u64] = if geom.rank == 1 { &[row] } else { &[row, cs] };
        let name = geom.track.name();
        let write_start = Instant::now();
        if !encoded.is_empty() {
            let shard = codec.assemble(encoded)?;
            // SAFETY: `assemble` lays the shard out with this array's own
            // sharding codecs, which are its only codecs.
            unsafe {
                geom.array
                    .store_encoded_chunk(indices, zarrs::storage::Bytes::from(shard))
            }
            .map_err(|e| PbzError::Store(format!("write {name}: {e}")))?;
            self.timings.write.record(write_start.elapsed());
        } else if parts.iter().any(|&(_, _, covered)| covered) {
            // Covered but every value is the fill: a store of fill erases.
            geom.array
                .erase_chunk(indices)
                .map_err(|e| PbzError::Store(format!("write {name}: {e}")))?;
        }
        for &(chunk_idx, cc, covered) in &parts {
            let chunk = self.chunk_range(chunk_idx as u64);
            self.record_close(geom, chunk, geom.buffer_columns(cc), covered);
        }
        Ok(())
    }
}

/// `WindowSink` fixes the error type, so an engine error crosses the reader
/// inside `ReaderError::Other`.
fn to_reader_err(err: PbzError) -> ReaderError {
    ReaderError::Other(anyhow::Error::new(err))
}

fn from_reader_err(err: ReaderError) -> PbzError {
    match err {
        ReaderError::Other(boxed) => boxed
            .downcast::<PbzError>()
            .unwrap_or_else(|boxed| PbzError::Reader(ReaderError::Other(boxed))),
        err => PbzError::Reader(err),
    }
}

/// Per-piece streaming state: one chunk of scratch per touch entry, merged
/// into the shared buffers each time the walk leaves a chunk.
struct PieceSink<'a, 'c> {
    ctx: &'a RunCtx<'c>,
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
    fn flat(&self, local: u64) -> u64 {
        self.window_flat_base + (local - self.window_local_base)
    }

    /// Drop this piece's claim on every buffer of `chunk_idx` and write the
    /// ones it finished. Called exactly once per chunk of the span, covered
    /// or not, so the counts stay exact.
    fn finish_chunk(&self, chunk_idx: u64, covered: bool) -> Result<()> {
        let chunk = self.ctx.chunk_range(chunk_idx);
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
            self.ctx.slots.lock().expect("slot map poisoned").remove(&(
                t_idx,
                chunk_idx as usize,
                cc,
            ));
            self.ctx
                .close_buffer(t_idx, cc, chunk_idx as usize, chunk.clone(), data, any_data)?;
        }
        Ok(())
    }

    /// Merge scratch rows `0..n` for the window at flat `flat_lo` into that
    /// chunk's buffers. Each lock covers one region copy, so pieces sharing a
    /// buffer stay concurrent through decode.
    fn merge(&self, flat_lo: u64, n: usize) -> Result<()> {
        let chunk_idx = flat_lo / self.ctx.chunk_len();
        let chunk = self.ctx.chunk_range(chunk_idx);
        let row0 = (flat_lo - chunk.start) as usize;
        let rows = (chunk.end - chunk.start) as usize;
        for (idx, entry) in self.entries.iter().enumerate() {
            let wait_start = Instant::now();
            let handle = self.ctx.slot((entry.t_idx, chunk_idx as usize, entry.cc));
            let mut guard = handle.inner.lock().expect("buffer slot poisoned");
            self.ctx.timings.slot_wait.record(wait_start.elapsed());
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
            self.ctx.timings.merge.record(merge_start.elapsed());
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
            for (&(field, _), sink) in entry.fields.iter().zip(local.column_sinks(0..n)) {
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
        let chunk_idx = flat_lo / self.ctx.chunk_len();
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
    let windows = contig_windows(ctx.reference, piece.start, piece.end);
    let entries = &ctx.plan.entries[piece.reader];
    let chunk_len = ctx.chunk_len();
    let first_chunk = piece.start / chunk_len;
    let mut sink = PieceSink {
        ctx,
        entries,
        scratch: Vec::new(),
        window_flat_base: 0,
        window_local_base: 0,
        current: None,
    };

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
            .map_err(from_reader_err)?;
    }
    ctx.timings.decode.record(read_start.elapsed());

    // Finish the last streamed chunk, then every chunk the stream never
    // reached, so an uncovered chunk inside a covered span still elides.
    let next = match sink.current.take() {
        Some((chunk_idx, chunk_covered)) => {
            sink.finish_chunk(chunk_idx, chunk_covered)?;
            chunk_idx + 1
        }
        None => first_chunk,
    };
    for chunk_idx in next..=(piece.end - 1) / chunk_len {
        sink.finish_chunk(chunk_idx, false)?;
    }
    Ok(())
}

fn build_touch_plan(
    routing: &ImportRouting,
    geoms: &[TargetGeom<'_>],
    n_readers: usize,
) -> TouchPlan {
    let mut slot_pieces: Vec<Vec<usize>> = geoms
        .iter()
        .map(|g| vec![0; g.width.div_ceil(g.col_chunk).max(1) as usize])
        .collect();
    let mut entries = Vec::with_capacity(n_readers);
    for reader in 0..n_readers {
        let mut touched: Vec<TouchEntry> = Vec::new();
        for (t_idx, geom) in geoms.iter().enumerate() {
            // Columns ascend, so the fields of one column chunk are adjacent.
            for (field, col) in routing.reader_columns(reader, t_idx) {
                let cc = col / geom.col_chunk;
                let local = (col % geom.col_chunk) as usize;
                match touched.last_mut() {
                    Some(last) if last.t_idx == t_idx && last.cc == cc => {
                        last.fields.push((field, local));
                    }
                    _ => {
                        slot_pieces[t_idx][cc as usize] += 1;
                        touched.push(TouchEntry {
                            t_idx,
                            cc,
                            fields: vec![(field, local)],
                        });
                    }
                }
            }
        }
        entries.push(touched);
    }
    let group_sizes = slot_pieces
        .iter()
        .map(|pieces| pieces.iter().filter(|&&n| n > 0).count())
        .collect();
    let col_shard_ccs = slot_pieces
        .iter()
        .zip(geoms)
        .map(|(pieces, geom)| {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            if geom.shard_aligned {
                for (cc, _) in pieces.iter().enumerate().filter(|(_, n)| **n > 0) {
                    *counts.entry(geom.col_shard(cc as u64)).or_insert(0) += 1;
                }
            }
            counts
        })
        .collect();
    TouchPlan {
        entries,
        slot_pieces,
        group_sizes,
        col_shard_ccs,
    }
}

pub(crate) fn validate_layout(tracks: &[Destination], routing: &ImportRouting) -> Result<()> {
    target_geometries(tracks, routing).map(|_| ())
}

fn target_geometries<'t>(
    tracks: &'t [Destination],
    routing: &ImportRouting,
) -> Result<Vec<TargetGeom<'t>>> {
    let mut geoms: Vec<TargetGeom<'_>> = Vec::with_capacity(tracks.len());
    for (track, target) in tracks.iter().zip(&routing.targets) {
        let array = track.values_array();
        let rank = track.rank();
        let width = track.columns_count()? as u64;
        let sharded = array.is_sharded();
        let write_unit = track.write_unit_shape()?;
        let (chunk_len, col_chunk) = if sharded {
            let sub = array.effective_subchunk_shape().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "track {:?}: sharded array with indeterminate subchunk shape",
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
        let shard_pos = write_unit[0] as u64;
        let shard_col = if rank == 2 { write_unit[1] as u64 } else { 1 };
        let shard_aligned = array.is_exclusively_sharded()
            && shard_pos.is_multiple_of(chunk_len)
            && shard_col.is_multiple_of(col_chunk)
            && target.columns.start.is_multiple_of(shard_col)
            && (target.columns.end.is_multiple_of(shard_col) || target.columns.end == width);
        let shard_codec = if shard_aligned {
            Some(ShardCodec::new(
                &array, rank, chunk_len, col_chunk, shard_pos, shard_col,
            )?)
        } else {
            None
        };
        geoms.push(TargetGeom {
            shard_codec,
            track,
            fill: array.fill_value().clone(),
            rank,
            dtype: track.dtype(),
            width,
            col_chunk,
            sharded,
            shard_pos,
            shard_col,
            shard_aligned,
            chunks_per_shard: if shard_aligned {
                shard_pos / chunk_len
            } else {
                1
            },
            columns: target.columns.clone(),
            array,
            array_chunk_len: chunk_len,
        });
    }

    Ok(geoms)
}

pub(crate) fn run_import<R: ValueReader>(
    readers: Vec<R>,
    tracks: &[Destination],
    routing: &ImportRouting,
    config: &Config,
    handle_budget: usize,
) -> Result<Report> {
    let n_readers = readers.len();

    let geoms = target_geometries(tracks, routing)?;

    let reference = Arc::clone(tracks[0].reference());
    let total = tracks[0].total_len();
    let workers = config.workers.max(1);
    let chunk_len = geoms[0].array_chunk_len.max(1);
    let n_chunks = usize::try_from(total.div_ceil(chunk_len))
        .map_err(|_| PbzError::Metadata("chunk count exceeds usize".into()))?;
    let decode_chunks = match config.decode_chunks {
        0 => auto_decode_chunks(total, chunk_len, n_readers, workers),
        n => n.clamp(1, n_chunks.max(1)),
    };
    let decode_chunks = snap_decode_chunks(decode_chunks, &geoms);
    let span_len = chunk_len * decode_chunks as u64;
    let n_spans = usize::try_from(total.div_ceil(span_len))
        .map_err(|_| PbzError::Metadata("span count exceeds usize".into()))?;

    let spans: Vec<(u64, u64)> = (0..n_spans as u64)
        .map(|i| (i * span_len, ((i + 1) * span_len).min(total)))
        .collect();

    if let Some(progress) = &config.progress {
        let total_bytes: u64 = geoms
            .iter()
            .map(|g| total * (g.columns.end - g.columns.start) * g.dtype.size() as u64)
            .sum();
        progress.set_total(total_bytes);
    }

    let plan = build_touch_plan(routing, &geoms, n_readers);

    let in_flight = auto_in_flight_spans(workers, n_readers).min(n_spans.max(1));
    let handle_budget = match handle_budget {
        0 => auto_handle_budget(soft_fd_limit(), n_readers),
        n => n,
    };
    let pool_size = (handle_budget / n_readers.max(1)).clamp(1, workers);
    let pieces = n_spans * n_readers;

    // Open buffers are bounded by the spans in flight; scratch by the
    // workers. Both are whole chunks.
    let chunk_row_bytes: u64 = geoms
        .iter()
        .map(|g| chunk_len * (g.columns.end - g.columns.start) * g.dtype.size() as u64)
        .sum();
    let scratch_chunk_bytes: u64 = plan
        .entries
        .iter()
        .map(|entries| {
            entries
                .iter()
                .map(|e| chunk_len * e.fields.len() as u64 * geoms[e.t_idx].dtype.size() as u64)
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

    let mut partial_opts = CodecOptions::default();
    partial_opts.set_experimental_partial_encoding(true);
    let ctx = RunCtx {
        slots: Mutex::new(HashMap::new()),
        gate: SpanGate::new(in_flight),
        span_remaining: (0..n_spans).map(|_| AtomicUsize::new(n_readers)).collect(),
        bytes_written: AtomicU64::new(0),
        chunks_written: AtomicUsize::new(0),
        chunks_skipped: AtomicUsize::new(0),
        timings: RunTimings::default(),
        err_flag: AtomicBool::new(false),
        first_err: Mutex::new(None),
        geoms: &geoms,
        reference: &reference,
        plan: &plan,
        progress: config.progress.as_deref(),
        cancel: config.cancel.as_deref(),
        partial_opts,
        shard_locks: Mutex::new(HashMap::new()),
        chunk_groups: Mutex::new(HashMap::new()),
        shard_groups: Mutex::new(HashMap::new()),
        total,
    };

    let pools: Vec<HandlePool<R>> = readers
        .into_iter()
        .map(|base| HandlePool::new(base, pool_size))
        .collect();

    let (piece_tx, piece_rx) = bounded::<Piece>((workers * 2).max(1));
    // One receiver per worker and none kept here, so a send fails once every
    // worker has exited.
    let receivers = Mutex::new(vec![piece_rx; workers]);

    // Decode workers and zarrs' subchunk encoding share this one pool, so
    // `workers` bounds the busy threads: a worker that writes a buffer helps
    // run its encode instead of idling while a second pool does the work.
    // `broadcast` runs exactly one worker loop per pool thread, so a thread
    // waiting on an encode never picks up a second, queued worker loop.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|i| format!("pbz-import-{i}"))
        .build()
        .map_err(|e| PbzError::Store(format!("build import thread pool: {e}")))?;

    let gate_wait = thread::scope(|scope| {
        let ctx = &ctx;
        let pools = &pools;
        let receivers = &receivers;
        let pool = &pool;
        scope.spawn(move || {
            pool.broadcast(|_| {
                let Some(piece_rx) = receivers
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .pop()
                else {
                    return;
                };
                let fail = |e: PbzError| {
                    ctx.record_err(e);
                    pools.iter().for_each(HandlePool::abort);
                };
                let mut busy = Duration::ZERO;
                let mut idle = Duration::ZERO;
                let run = panic::catch_unwind(AssertUnwindSafe(|| {
                    loop {
                        let wait_start = Instant::now();
                        let Ok(piece) = piece_rx.recv() else { break };
                        idle += wait_start.elapsed();
                        let work_start = Instant::now();
                        // Stalling on a pool is not work, so it comes back out of
                        // `busy` below and is reported as its own stage.
                        let mut stalled = Duration::ZERO;
                        'piece: {
                            if ctx.has_err() {
                                // Drain so the channel closes cleanly; the gate is
                                // already aborted, so nothing blocks on us.
                                break 'piece;
                            }
                            if ctx.check_cancel() {
                                pools.iter().for_each(HandlePool::abort);
                                break 'piece;
                            }
                            let checkout_start = Instant::now();
                            let checked_out = pools[piece.reader].checkout();
                            stalled = checkout_start.elapsed();
                            ctx.timings.handle_wait.record(stalled);
                            let mut reader = match checked_out {
                                Ok(Some(reader)) => reader,
                                Ok(None) => break 'piece,
                                Err(e) => {
                                    fail(e);
                                    break 'piece;
                                }
                            };
                            let result = process_piece(ctx, &mut reader, piece);
                            pools[piece.reader].checkin(reader);
                            if let Err(e) = result {
                                fail(e);
                                break 'piece;
                            }
                            if ctx.span_remaining[piece.span].fetch_sub(1, Ordering::AcqRel) == 1 {
                                ctx.gate.release();
                            }
                        }
                        busy += work_start.elapsed().saturating_sub(stalled);
                    }
                }));
                // A panic loses this worker's handle and span count, so the
                // gate and every pool must abort or the others could wait
                // forever.
                if let Err(payload) = run {
                    fail(PbzError::Metadata(format!(
                        "import worker panicked: {}",
                        panic_message(&*payload)
                    )));
                }
                drop(piece_rx);
                ctx.timings.worker_busy.record(busy);
                ctx.timings.worker_idle.record(idle);
            });
        });

        let mut gate_wait = Duration::ZERO;
        'produce: for (span_idx, &(start, end)) in spans.iter().enumerate() {
            if ctx.has_err() {
                break;
            }
            if ctx.check_cancel() {
                pools.iter().for_each(HandlePool::abort);
                break;
            }
            let gate_start = Instant::now();
            ctx.gate.acquire();
            gate_wait += gate_start.elapsed();
            if ctx.has_err() {
                break;
            }
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
        gate_wait
    });

    if let Some(progress) = &config.progress {
        progress.done();
    }

    let completed = ctx.chunks_written.load(Ordering::Relaxed);
    let skipped = ctx.chunks_skipped.load(Ordering::Relaxed);
    let bytes = ctx.bytes_written.load(Ordering::Relaxed);
    let wall = started.elapsed();
    info!(
        "import pipeline finished: {completed} buffer(s) written, {skipped} elided, \
         {bytes} bytes in {wall:.1?}"
    );
    if completed == 0 && skipped > 0 {
        warn!(
            "import pipeline wrote no data: all {skipped} buffers were elided because \
             no source covered them"
        );
    }

    ctx.timings.log(wall, workers, gate_wait);

    let first_err = ctx
        .first_err
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    if let Some(e) = first_err {
        return Err(e);
    }

    // Every piece drops its claim on every chunk of its span, so a clean run
    // ends with nothing open. A reader that skipped a window would otherwise
    // leave that chunk unwritten, uncounted, and unreported.
    let open = ctx.slots.lock().expect("slot map poisoned").len()
        + ctx
            .chunk_groups
            .lock()
            .expect("chunk group map poisoned")
            .len()
        + ctx
            .shard_groups
            .lock()
            .expect("shard group map poisoned")
            .len();
    if open > 0 {
        return Err(PbzError::Metadata(format!(
            "engine invariant: {open} chunk buffer(s) never closed; a reader did not \
             report every window of its span"
        )));
    }

    Ok(Report {
        positions: total,
        chunks_written: completed,
        chunks_skipped: skipped,
        bytes_written: bytes,
        wall_seconds: wall.as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use ndarray::{Ix1, Ix2};
    use tempfile::TempDir;

    use super::*;
    use crate::Contig;
    use crate::TrackConfig;
    use crate::dataset::{Dataset, Reference};
    use crate::import::{Import, ImportBuilder};
    use crate::io::{OutputSchema, ValueReader};

    /// Synthetic single-field reader over one contig: position `p` inside
    /// `covered` (or `extra`) reads as `base + p`; everything else is left
    /// untouched.
    #[derive(Clone)]
    struct SynthReader {
        genome: Reference,
        schema: OutputSchema,
        covered: Range<u64>,
        extra: Option<Range<u64>>,
        /// Reported as covered, but no value is written: the window keeps
        /// the fill.
        fill_cover: Option<Range<u64>>,
        base: i64,
    }

    impl SynthReader {
        fn new(genome: Reference, dtype: Dtype, covered: Range<u64>, base: i64) -> Self {
            Self {
                genome,
                schema: OutputSchema::single("value", dtype),
                covered,
                extra: None,
                fill_cover: None,
                base,
            }
        }

        /// A second covered range, disjoint from the first.
        fn with_extra_cover(mut self, extra: Range<u64>) -> Self {
            self.extra = Some(extra);
            self
        }

        fn with_fill_cover(mut self, range: Range<u64>) -> Self {
            self.fill_cover = Some(range);
            self
        }

        fn ranges(&self) -> impl Iterator<Item = &Range<u64>> {
            std::iter::once(&self.covered).chain(self.extra.iter())
        }
    }

    impl ValueReader for SynthReader {
        fn contigs(&self) -> &Reference {
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

        /// Reports a window as uncovered when no range reaches it, as a
        /// sparse source does.
        fn read_windows(
            &mut self,
            contig: &str,
            start: u64,
            end: u64,
            cuts: &[u64],
            sink: &mut dyn WindowSink,
        ) -> std::result::Result<(), crate::io::ReaderError> {
            let mut lo = start;
            for &hi in cuts.iter().chain(std::iter::once(&end)) {
                let covered = self
                    .ranges()
                    .chain(self.fill_cover.iter())
                    .any(|r| lo < r.end && hi > r.start);
                if covered {
                    let mut outputs = sink.sinks(lo, hi)?;
                    self.read_into(contig, lo, hi, &mut outputs)?;
                }
                sink.done(lo, hi, covered)?;
                lo = hi;
            }
            Ok(())
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(self.clone())
        }
    }

    /// Delegates to a reader and records, per read, whether it ran on a
    /// rayon pool thread and that pool's size.
    #[derive(Clone)]
    struct ThreadProbe<R> {
        inner: R,
        seen: Arc<std::sync::Mutex<Vec<(bool, usize)>>>,
    }

    impl<R: ValueReader> ValueReader for ThreadProbe<R> {
        fn contigs(&self) -> &Reference {
            self.inner.contigs()
        }

        fn output_schema(&self) -> &OutputSchema {
            self.inner.output_schema()
        }

        fn read_into(
            &mut self,
            contig: &str,
            start: u64,
            end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            self.seen.lock().unwrap().push((
                rayon::current_thread_index().is_some(),
                rayon::current_num_threads(),
            ));
            self.inner.read_into(contig, start, end, outputs)
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(Self {
                inner: self.inner.fork()?,
                seen: Arc::clone(&self.seen),
            })
        }
    }

    fn one_contig(len: u64) -> Reference {
        Reference::new(vec![Contig {
            name: "chr1".into(),
            length: len,
        }])
        .unwrap()
    }

    /// An unsharded layout of `chunk` positions by two columns per chunk.
    fn column_pair_chunks(chunk: u64) -> crate::ExplicitArraySpec {
        crate::ExplicitArraySpec::from_value(serde_json::json!({
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [chunk, 2]}},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
        }))
        .unwrap()
    }

    fn dataset(path: &Path, genome: &Reference) -> Dataset {
        Dataset::create(path, genome.clone()).unwrap()
    }

    /// Import into a new track. A config with columns takes one reader per
    /// column.
    fn import<'t, R: ValueReader>(
        ds: &'t mut Dataset,
        name: &str,
        cfg: TrackConfig,
        readers: Vec<R>,
    ) -> ImportBuilder<'t, R> {
        let labels = cfg.columns.clone();
        let builder = Import::from_readers(readers)
            .unwrap()
            .into_dataset(ds, vec![(name.into(), cfg)]);
        match labels {
            Some(labels) => builder.readers_as_columns().expect_column_labels(labels),
            None => builder,
        }
    }

    fn read_contig<T: zarrs::array::ElementOwned>(
        ds: &Dataset,
        name: &str,
        contig: &str,
    ) -> ndarray::ArrayD<T> {
        ds.read_region::<T>(name, &contig.parse().unwrap()).unwrap()
    }

    /// Number of chunk files (anything but `zarr.json`) under an array dir.
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
            let mut ds = dataset(&dir.join(name), &genome);
            import(&mut ds, "v", cfg.columns(labels.clone()), readers())
                .config(Config {
                    workers: 3,
                    decode_chunks,
                    ..Config::default()
                })
                .run()
                .unwrap();
            ds
        };
        let sharded = || {
            TrackConfig::new(Dtype::I32)
                .chunk(16)
                .shard(3)
                .shard_columns(2)
        };
        let dir = TempDir::new().unwrap();
        let a = run(dir.path(), "a.pbz", sharded(), 3);
        let b = run(dir.path(), "b.pbz", sharded(), 1);
        let c = run(
            dir.path(),
            "c.pbz",
            TrackConfig::new(Dtype::I32).chunk(16).unsharded(),
            3,
        );
        let read = |ds: &Dataset| -> ndarray::Array2<i32> {
            let values = read_contig::<i32>(ds, "v", "chr1");
            assert_eq!(values.shape()[0], len as usize);
            values.into_dimensionality::<Ix2>().unwrap()
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
        let mut ds = dataset(&dir.path().join("elide.pbz"), &genome);
        // Covered: chunk 0 (0..16) and chunk 3 (48..64); chunks 1 and 2 empty.
        let reader =
            SynthReader::new(genome.clone(), Dtype::I32, 0..16, 7).with_extra_cover(48..64);
        let cfg = TrackConfig::new(Dtype::I32).chunk(16).unsharded();
        let report = import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 1,
                decode_chunks: 4,
                ..Config::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.chunks_skipped, 2);
        assert_eq!(chunk_file_count(&dir.path().join("elide.pbz/v")), 2);
    }

    /// A shard-aligned target writes one file per covered shard and no file
    /// for an uncovered one; the tail shard pads with fill and reads back
    /// clipped. Uncovered positions read back as the custom fill.
    #[test]
    fn whole_shard_write_and_elision() {
        let dir = TempDir::new().unwrap();
        let len = 80u64;
        let genome = one_contig(len);
        let mut ds = dataset(&dir.path().join("shard.pbz"), &genome);
        let cfg = TrackConfig::new(Dtype::I32)
            .chunk(16)
            .shard(3)
            .fill_value(serde_json::json!(-1));
        // Shard row 0 holds chunks 0..3 (0..48); row 1 holds chunks 3..5
        // (48..80). Coverage 0..40 leaves row 1 untouched.
        let reader = SynthReader::new(genome.clone(), Dtype::I32, 0..40, 7);
        let report = import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 2,
                decode_chunks: 2,
                ..Config::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.chunks_written, 3);
        assert_eq!(report.chunks_skipped, 2);
        assert_eq!(chunk_file_count(&dir.path().join("shard.pbz/v")), 1);
        let values = read_contig::<i32>(&ds, "v", "chr1")
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert_eq!(values.len(), len as usize);
        for p in 0..40 {
            assert_eq!(values[p], 7 + p as i32);
        }
        for p in 40..len as usize {
            assert_eq!(values[p], -1);
        }
    }

    #[derive(Default)]
    struct CountingSink {
        total: AtomicU64,
        ticked: AtomicU64,
    }

    impl ProgressSink for CountingSink {
        fn set_total(&self, bytes: u64) {
            self.total.store(bytes, Ordering::Relaxed);
        }
        fn tick(&self, bytes: u64) {
            self.ticked.fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// Skipped buffers tick too, so the progress count ends at its total on
    /// unsharded, sharded and sharded cohort tracks.
    #[test]
    fn progress_reaches_total_with_skipped_buffers() {
        let len = 80u64;
        let genome = one_contig(len);
        let labels: Vec<String> = (0..3).map(|i| format!("s{i}")).collect();
        let cases = [
            TrackConfig::new(Dtype::I32).chunk(16).unsharded(),
            TrackConfig::new(Dtype::I32).chunk(16).shard(3),
            TrackConfig::new(Dtype::I32)
                .columns(labels.clone())
                .chunk(16)
                .shard(3)
                .shard_columns(2),
        ];
        let dir = TempDir::new().unwrap();
        for (i, cfg) in cases.into_iter().enumerate() {
            let n = if i == 2 { labels.len() } else { 1 };
            let readers: Vec<SynthReader> = (0..n)
                .map(|_| SynthReader::new(genome.clone(), Dtype::I32, 0..20, 7))
                .collect();
            let mut ds = dataset(&dir.path().join(format!("p{i}.pbz")), &genome);
            let sink = Arc::new(CountingSink::default());
            let report = import(&mut ds, "v", cfg, readers)
                .config(Config {
                    workers: 2,
                    decode_chunks: 1,
                    progress: Some(sink.clone()),
                    ..Config::default()
                })
                .run()
                .unwrap();
            assert!(report.chunks_skipped > 0, "case {i}");
            let total = sink.total.load(Ordering::Relaxed);
            assert_eq!(total, len * n as u64 * 4, "case {i}");
            assert_eq!(sink.ticked.load(Ordering::Relaxed), total, "case {i}");
        }
    }

    /// A decode span crossing a contig boundary mid-chunk: the split chunk
    /// takes two windows from one piece, closes once, and keeps each contig's
    /// rows in place.
    #[test]
    fn span_crossing_contig_boundary_places_both_contigs() {
        let dir = TempDir::new().unwrap();
        let genome = Reference::new(vec![
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
        let mut ds = dataset(&dir.path().join("split.pbz"), &genome);
        let reader = SynthReader::new(genome, Dtype::I32, 0..22, 100);
        let cfg = TrackConfig::new(Dtype::I32).chunk(8).unsharded();
        let report = import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 1,
                decode_chunks: 4,
                ..Config::default()
            })
            .run()
            .unwrap();
        assert_eq!(report.chunks_written, 4);
        assert_eq!(report.chunks_skipped, 0);

        for (contig, len) in [("chr1", 10u64), ("chr2", 22u64)] {
            let values = read_contig::<i32>(&ds, "v", contig)
                .into_dimensionality::<Ix1>()
                .unwrap();
            assert_eq!(values.len(), len as usize);
            for pos in 0..len as usize {
                assert_eq!(values[pos], 100 + pos as i32, "{contig} position {pos}");
            }
        }
    }

    #[test]
    fn workers_run_on_a_pool_of_exactly_the_requested_size() {
        // Decode and zarrs encode share one pool of `workers` threads, so
        // `--threads N` bounds the busy threads at N rather than 2N.
        let dir = TempDir::new().unwrap();
        let genome = one_contig(400);
        let mut ds = dataset(&dir.path().join("pool.pbz"), &genome);
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let reader = ThreadProbe {
            inner: SynthReader::new(genome, Dtype::I32, 0..400, 0),
            seen: Arc::clone(&seen),
        };
        let cfg = TrackConfig::new(Dtype::I32).chunk(16).unsharded();
        import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 3,
                decode_chunks: 1,
                ..Config::default()
            })
            .run()
            .unwrap();
        let seen = seen.lock().unwrap();
        assert!(!seen.is_empty());
        assert!(
            seen.iter().all(|&probe| probe == (true, 3)),
            "reads ran outside a 3-thread pool: {seen:?}"
        );
    }

    #[test]
    fn nan_fill_prefill_and_elision() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(40);
        let mut ds = dataset(&dir.path().join("fill.pbz"), &genome);
        let reader = SynthReader::new(genome, Dtype::F32, 16..26, 0);
        // F32 tracks default to a NaN fill.
        let cfg = TrackConfig::new(Dtype::F32).chunk(16).unsharded();
        let report = import(&mut ds, "signal", cfg, vec![reader]).run().unwrap();
        // Chunks 16,16,8: only the middle has coverage.
        assert_eq!(report.chunks_written, 1);
        assert_eq!(report.chunks_skipped, 2);

        let values = read_contig::<f32>(&ds, "signal", "chr1")
            .into_dimensionality::<Ix1>()
            .unwrap();
        for pos in 16..26 {
            assert_eq!(values[pos], pos as f32);
        }
        // Uncovered positions of the WRITTEN buffer (26..32) and of the
        // ELIDED buffers (0..16, 32..40) both read back as the NaN fill.
        for pos in (0..16).chain(26..40) {
            assert!(values[pos].is_nan(), "position {pos} should be NaN");
        }
        assert_eq!(chunk_file_count(&dir.path().join("fill.pbz/signal")), 1);
    }

    /// Reader whose `read_into` records how many decodes run at once. Each
    /// call waits (bounded) for a peer, so serialized decodes stay at 1.
    #[derive(Clone)]
    struct ConcurrencyProbe {
        genome: Reference,
        schema: OutputSchema,
        /// `(currently decoding, max observed)`.
        gauge: Arc<(Mutex<(usize, usize)>, Condvar)>,
    }

    impl ValueReader for ConcurrencyProbe {
        fn contigs(&self) -> &Reference {
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
        let mut ds = dataset(&dir.path().join("conc.pbz"), &genome);
        let cfg = TrackConfig::new(Dtype::I32)
            .columns(labels)
            .codecs(column_pair_chunks(16));

        let gauge = Arc::new((Mutex::new((0usize, 0usize)), Condvar::new()));
        let readers: Vec<ConcurrencyProbe> = (0..2)
            .map(|_| ConcurrencyProbe {
                genome: genome.clone(),
                schema: OutputSchema::single("value", Dtype::I32),
                gauge: Arc::clone(&gauge),
            })
            .collect();

        import(&mut ds, "depth", cfg, readers)
            .config(Config {
                workers: 2,
                ..Config::default()
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
        let plain_cfg = TrackConfig::new(Dtype::I32)
            .columns(labels.clone())
            .chunk(16)
            .unsharded();
        let sharded_cfg = plain_cfg.clone().shard(2).shard_columns(4);
        let mut plain = dataset(&dir.path().join("plain.pbz"), &genome);
        let mut sharded = dataset(&dir.path().join("sharded.pbz"), &genome);

        for (ds, cfg) in [(&mut plain, plain_cfg), (&mut sharded, sharded_cfg)] {
            let report = import(ds, "depth", cfg, readers())
                .config(Config {
                    workers: 4,
                    ..Config::default()
                })
                .run()
                .unwrap();
            assert!(report.chunks_written > 0);
        }

        let plain_data = read_contig::<i32>(&plain, "depth", "chr1")
            .into_dimensionality::<Ix2>()
            .unwrap();
        let sharded_data = read_contig::<i32>(&sharded, "depth", "chr1")
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(plain_data.nrows(), len as usize);
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
        let storage = zarrs::filesystem::FilesystemStore::new(dir.path().join("sharded.pbz"));
        let array = Array::open(Arc::new(storage.unwrap()), "/depth").unwrap();
        assert_eq!(array.chunk_shape_usize(&[0, 0]).unwrap(), vec![32, 4]);
        assert!(array.is_exclusively_sharded());
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
        let plain_cfg = TrackConfig::new(Dtype::I32)
            .columns(labels.clone())
            .chunk(16)
            .unsharded();
        let sharded_cfg = plain_cfg.clone().shard(2).shard_columns(2);
        let mut plain = dataset(&dir.path().join("plain.pbz"), &genome);
        let mut sharded = dataset(&dir.path().join("sharded.pbz"), &genome);

        for (ds, cfg) in [(&mut plain, plain_cfg), (&mut sharded, sharded_cfg)] {
            let report = import(ds, "depth", cfg, readers())
                .config(Config {
                    workers: 4,
                    ..Config::default()
                })
                .run()
                .unwrap();
            // 6 chunks x 4 column chunks; chunks 64..96 have no coverage.
            assert_eq!(report.chunks_written, 16);
            assert_eq!(report.chunks_skipped, 8);
        }

        let plain_data = read_contig::<i32>(&plain, "depth", "chr1");
        let sharded_data = read_contig::<i32>(&sharded, "depth", "chr1");
        assert_eq!(plain_data.shape(), [len as usize, 4]);
        assert_eq!(plain_data, sharded_data);

        // 3 x 2 shard grid; the uncovered position shard (64..96) elides, so
        // its two shard files are never created.
        assert_eq!(chunk_file_count(&dir.path().join("sharded.pbz/depth")), 4);
    }

    /// One-column inner chunks in shards that the import encodes chunk by
    /// chunk: the values and the elided inner chunks match an unsharded
    /// import. The geometry has a partial tail shard on both axes, a partial
    /// tail chunk, and a covered chunk that holds only the fill.
    #[test]
    fn encoded_shards_match_unsharded_chunks() {
        let len = 100u64;
        let genome = one_contig(len);
        let labels: Vec<String> = (0..3).map(|i| format!("s{i}")).collect();
        let readers = || -> Vec<SynthReader> {
            vec![
                SynthReader::new(genome.clone(), Dtype::I32, 5..60, 1000),
                SynthReader::new(genome.clone(), Dtype::I32, 20..40, 2000).with_fill_cover(64..80),
                SynthReader::new(genome.clone(), Dtype::I32, 90..100, 3000),
            ]
        };
        let dir = TempDir::new().unwrap();
        let plain_cfg = TrackConfig::new(Dtype::I32)
            .columns(labels.clone())
            .chunk(16)
            .unsharded();
        let sharded_cfg = plain_cfg.clone().shard(2).shard_columns(2);
        let mut plain = dataset(&dir.path().join("plain.pbz"), &genome);
        let mut sharded = dataset(&dir.path().join("sharded.pbz"), &genome);
        for (ds, cfg) in [(&mut plain, plain_cfg), (&mut sharded, sharded_cfg)] {
            import(ds, "depth", cfg, readers())
                .config(Config {
                    workers: 4,
                    decode_chunks: 1,
                    ..Config::default()
                })
                .run()
                .unwrap();
        }
        let plain_data = read_contig::<i32>(&plain, "depth", "chr1");
        assert_eq!(plain_data.shape(), [len as usize, 3]);
        assert_eq!(plain_data, read_contig::<i32>(&sharded, "depth", "chr1"));
        assert_eq!(plain_data[[70, 1]], 0);
        assert_eq!(plain_data[[99, 2]], 3099);

        let storage = Arc::new(
            zarrs::filesystem::FilesystemStore::new(dir.path().join("sharded.pbz")).unwrap(),
        );
        let array = Array::open(storage, "/depth").unwrap();
        let ArrayMetadata::V3(meta) = array.metadata() else {
            panic!("expected v3 metadata")
        };
        let config = meta.codecs[0]
            .to_typed_configuration::<ShardingCodecConfiguration>()
            .unwrap();
        let codec = zarrs::array::codec::ShardingCodec::new_with_configuration(&config).unwrap();
        let cps = [NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()];
        let mut stored = 0;
        for chunk in 0..7u64 {
            for column in 0..3u64 {
                let plain_file = dir
                    .path()
                    .join(format!("plain.pbz/depth/c/{chunk}/{column}"));
                let shard_file =
                    dir.path()
                        .join(format!("sharded.pbz/depth/c/{}/{}", chunk / 2, column / 2));
                let in_shard = std::fs::read(&shard_file).is_ok_and(|bytes| {
                    let index = codec
                        .decode_index(&bytes, &cps, &CodecOptions::default())
                        .unwrap();
                    let i = ((chunk % 2) * 2 + column % 2) as usize;
                    index[2 * i] != u64::MAX
                });
                assert_eq!(
                    plain_file.exists(),
                    in_shard,
                    "chunk {chunk} column {column}"
                );
                stored += usize::from(in_shard);
            }
        }
        // Column 0: chunks 0..4; column 1: chunks 1, 2; column 2: chunks 5, 6.
        assert_eq!(stored, 8);
        // Shard (2, 0) has only the fill-only chunk, so it has no file.
        assert!(!dir.path().join("sharded.pbz/depth/c/2/0").exists());
    }

    /// Counts live handles of one reader, base included, so a test can assert
    /// the pool cap.
    struct ForkCounter {
        genome: Reference,
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
        fn contigs(&self) -> &Reference {
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

    /// One reader, many workers, a budget of two handles: live handles, base
    /// included, never exceed the budget, and every piece completes.
    #[test]
    fn handle_pool_caps_forks_and_completes() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(64);
        let mut ds = dataset(&dir.path().join("pool.pbz"), &genome);

        let live = Arc::new(AtomicUsize::new(1));
        let peak = Arc::new(AtomicUsize::new(1));
        let reader = ForkCounter {
            genome: genome.clone(),
            schema: OutputSchema::single("value", Dtype::I32),
            live: Arc::clone(&live),
            peak: Arc::clone(&peak),
        };
        let cfg = TrackConfig::new(Dtype::I32).chunk(4).unsharded();
        let report = import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 8,
                ..Config::default()
            })
            .handle_budget(2)
            .run()
            .unwrap();
        let peak = peak.load(Ordering::SeqCst);
        assert!(peak <= 2, "peak handles {peak}");
        assert_eq!(report.chunks_written, 16);

        let values = read_contig::<i32>(&ds, "v", "chr1");
        assert_eq!(values.len(), 64);
        assert!(values.iter().all(|&x| x == 1));
    }

    /// Budget below the reader count: every reader still gets one handle.
    #[test]
    fn handle_budget_floor_finishes_import() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(32);
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut ds = dataset(&dir.path().join("floor.pbz"), &genome);
        let cfg = TrackConfig::new(Dtype::I32)
            .columns(labels)
            .chunk(8)
            .unsharded();

        let readers: Vec<SynthReader> = (0..3)
            .map(|i| SynthReader::new(genome.clone(), Dtype::I32, 0..32, (i + 1) * 100))
            .collect();
        import(&mut ds, "v", cfg, readers)
            .config(Config {
                workers: 4,
                ..Config::default()
            })
            .handle_budget(1)
            .run()
            .unwrap();

        let values = read_contig::<i32>(&ds, "v", "chr1")
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert_eq!(values[[5, 2]], 300 + 5);
    }

    /// Panics on any read that reaches past `panic_at`.
    #[derive(Clone)]
    struct PanicReader {
        genome: Reference,
        schema: OutputSchema,
        panic_at: u64,
    }

    impl ValueReader for PanicReader {
        fn contigs(&self) -> &Reference {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            _start: u64,
            end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            if end > self.panic_at {
                panic!("test reader panic");
            }
            outputs[0].as_i32_mut()?.fill(1);
            Ok(())
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(self.clone())
        }
    }

    /// One handle for many workers: the panicking worker never checks its
    /// handle in, and the import still returns an error.
    #[test]
    fn worker_panic_fails_the_import() {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let dir = TempDir::new().unwrap();
            let genome = one_contig(64);
            let mut ds = dataset(&dir.path().join("panic.pbz"), &genome);
            let reader = PanicReader {
                genome: genome.clone(),
                schema: OutputSchema::single("value", Dtype::I32),
                panic_at: 32,
            };
            let cfg = TrackConfig::new(Dtype::I32).chunk(4).unsharded();
            let result = import(&mut ds, "v", cfg, vec![reader])
                .config(Config {
                    workers: 4,
                    decode_chunks: 1,
                    ..Config::default()
                })
                .handle_budget(1)
                .run();
            let _ = tx.send(result.map(|_| ()).map_err(|e| e.to_string()));
        });
        let result = rx
            .recv_timeout(Duration::from_secs(60))
            .expect("import did not return after a worker panic");
        let err = result.unwrap_err();
        assert!(err.contains("panicked"), "{err}");
    }

    /// Sets `cancel` on its first read past `cancel_at` and counts reads.
    #[derive(Clone)]
    struct CancelReader {
        genome: Reference,
        schema: OutputSchema,
        cancel_at: u64,
        cancel: Arc<AtomicBool>,
        reads: Arc<AtomicUsize>,
    }

    impl ValueReader for CancelReader {
        fn contigs(&self) -> &Reference {
            &self.genome
        }

        fn output_schema(&self) -> &OutputSchema {
            &self.schema
        }

        fn read_into(
            &mut self,
            _contig: &str,
            _start: u64,
            end: u64,
            outputs: &mut [OutputSinkMut<'_>],
        ) -> std::result::Result<(), crate::io::ReaderError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if end > self.cancel_at {
                self.cancel.store(true, Ordering::SeqCst);
            }
            outputs[0].as_i32_mut()?.fill(1);
            Ok(())
        }

        fn fork(&self) -> std::result::Result<Self, crate::io::ReaderError> {
            Ok(self.clone())
        }
    }

    #[test]
    fn cancel_flag_stops_the_import_with_an_error() {
        let dir = TempDir::new().unwrap();
        let genome = one_contig(1024);
        let mut ds = dataset(&dir.path().join("cancel.pbz"), &genome);
        let cancel = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(AtomicUsize::new(0));
        let reader = CancelReader {
            genome: genome.clone(),
            schema: OutputSchema::single("value", Dtype::I32),
            cancel_at: 32,
            cancel: Arc::clone(&cancel),
            reads: Arc::clone(&reads),
        };
        let cfg = TrackConfig::new(Dtype::I32).chunk(4).unsharded();
        let result = import(&mut ds, "v", cfg, vec![reader])
            .config(Config {
                workers: 2,
                decode_chunks: 1,
                cancel: Some(cancel),
                ..Config::default()
            })
            .run();
        assert!(matches!(result, Err(PbzError::Cancelled)));
        let reads = reads.load(Ordering::SeqCst);
        assert!(reads < 256, "{reads} reads after cancel");
    }
}
