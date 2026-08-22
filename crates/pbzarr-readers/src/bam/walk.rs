//! Per-record CIGAR walk: turns one [`AlignedRead`] into depth and/or
//! per-base composition increments over a flat accumulator window.
//!
//! Claim semantics: an overlapping pair's claims (see [`super::mate::Claims`])
//! are split by contribution class, not tracked per field and not pooled into
//! one bitmap. A position's *base* claim means "this pair already contributed
//! depth here" (a BQ-passing Match base, or a Del position when
//! `count_deletions` makes deletions cover); it gates the second mate's depth
//! and a/c/g/t/n increments. A position's *event* claim means "this pair
//! already recorded a del/ref_skip/ins event here"; it gates only those three
//! fields.
//!
//! The split is what keeps Composition-mode depth equal to Depth-mode depth.
//! Mates disagree on op class inside their overlap often enough to matter
//! (one deletes or splices where the other matches); with a single bitmap the
//! first mate's event claimed the position outright and the second mate's
//! Match was gated wholesale, dropping depth and the base tallies in
//! Composition mode only.
//!
//! The three event fields (del, ref_skip, ins) share one event bitmap rather
//! than getting one each. That is deliberately coarse -- a pair whose mates
//! record *different* event classes at the same position counts only the
//! first -- and accepted: two mates of one molecule disagreeing on event class
//! at the same reference position is vanishingly rare, and counting an event
//! once at an already-claimed position is the failure mode overlap handling
//! exists to prevent.

use super::mate::{ClaimHandle, Claims, MateBuffer, Probe};
use super::record::{AlignedRead, CigarKind};
use super::{DepthFilter, OverlapMode};

/// Composition field order. Depth mode only ever touches index 0.
pub const FIELDS: [&str; 9] = ["depth", "a", "c", "g", "t", "n", "ins", "del", "ref_skip"];

const DEPTH: usize = 0;
const A: usize = 1;
const C: usize = 2;
const G: usize = 3;
const T: usize = 4;
const N: usize = 5;
const INS: usize = 6;
const DEL: usize = 7;
const REF_SKIP: usize = 8;

/// Which fields one import pass writes: depth-only (track `columns()` has one
/// `I32` field), or the full per-base composition breakdown (nine fields, in
/// [`FIELDS`] order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportMode {
    /// Depth only: one `I32` field per source.
    Depth,
    /// Depth plus per-base composition: `FIELDS.len()` `I32` fields per
    /// source (depth, a, c, g, t, n, ins, del, ref_skip).
    Composition,
}

impl ImportMode {
    /// Number of accumulator fields this mode writes.
    pub fn n_fields(self) -> usize {
        match self {
            ImportMode::Depth => 1,
            ImportMode::Composition => FIELDS.len(),
        }
    }
}

/// Contiguous per-field scratch for one walk window, `(n_fields, window_len)`.
/// Depth mode allocates a single field; composition mode allocates all of
/// [`FIELDS`]. Reused across windows via [`Accumulators::reset`].
pub struct Accumulators {
    pub fields: Vec<Vec<i32>>,
}

impl Accumulators {
    pub fn new(n_fields: usize, window_len: usize) -> Self {
        Self {
            fields: vec![vec![0i32; window_len]; n_fields],
        }
    }

    pub fn reset(&mut self, window_len: usize) {
        for field in &mut self.fields {
            field.clear();
            field.resize(window_len, 0);
        }
    }
}

/// The no-gate, depth-only Match-block inner loop for a `min_bq` above zero:
/// bumps `depth[i]` for every base whose quality clears it. A `min_bq` of
/// zero takes the unconditional block increment in `walk_record` instead and
/// never calls this. Kept as a standalone function so
/// it can be SIMD-optimized without touching the gated/composition paths,
/// which still need the per-base branch for claim recording and base
/// tallies. The 16-lane `wide::u8x16` body adds the 0/1 mask onto the `i32`
/// depth slice scalar-wise rather than widening, since this crate's `wide`
/// has no direct u8x16->i32x16 path.
pub fn bump_depth(depth: &mut [i32], quals: &[u8], min_bq: u8) {
    use wide::u8x16;

    let len = depth.len().min(quals.len());
    let chunks = len / 16;
    let min_bq_v = u8x16::splat(min_bq);

    for i in 0..chunks {
        let base = i * 16;
        let mut qual_arr = [0u8; 16];
        qual_arr.copy_from_slice(&quals[base..base + 16]);
        let mask = u8x16::from(qual_arr).simd_ge(min_bq_v) & u8x16::splat(1);
        let mask_arr = mask.to_array();
        for lane in 0..16 {
            depth[base + lane] += mask_arr[lane] as i32;
        }
    }

    let tail = chunks * 16;
    for i in tail..len {
        if quals[i] >= min_bq {
            depth[i] += 1;
        }
    }
}

/// A record may carry no per-base qualities at all (SAM `QUAL = *`, and
/// likewise `SEQ = *`), which is legal alongside a real CIGAR. Absent quality
/// means "unknown", and the `min_bq` gate is a lower bound on *known* quality,
/// so an absent score passes (htslib spells the same thing as 0xFF).
fn qual_at(read: &AlignedRead, read_idx: usize) -> u8 {
    read.qual.get(read_idx).copied().unwrap_or(u8::MAX)
}

/// Walk one record's CIGAR against `[win_start, win_end)`, incrementing
/// `acc` per the filter and mode. Positions outside the window are clipped
/// away at each op; positions already claimed by an overlapping mate (per
/// `filter.overlap`, see [`OverlapMode`]) are skipped.
pub fn walk_record(
    read: &AlignedRead,
    win_start: u64,
    win_end: u64,
    filter: &DepthFilter,
    mode: ImportMode,
    acc: &mut Accumulators,
    mates: &mut MateBuffer,
) {
    if read.flags & filter.exclude_flags != 0 {
        return;
    }
    if read.mapq < filter.min_mapq {
        return;
    }

    let probe = if filter.overlap == OverlapMode::None {
        Probe::Alone
    } else {
        mates.probe(read, filter.overlap)
    };

    enum State {
        None,
        Claiming(ClaimHandle),
        Gated(Claims),
    }
    impl State {
        fn base_blocked(&self, local: usize) -> bool {
            matches!(self, State::Gated(claims) if claims.base_claimed(local))
        }

        fn event_blocked(&self, local: usize) -> bool {
            matches!(self, State::Gated(claims) if claims.event_claimed(local))
        }

        /// No-op unless this record is the first of an overlapping pair
        /// (`Claiming`); a lone or already-gated record has nothing to claim.
        fn claim_base(&mut self, local: usize) {
            if let State::Claiming(handle) = self {
                handle.claim_base(local);
            }
        }

        fn claim_event(&mut self, local: usize) {
            if let State::Claiming(handle) = self {
                handle.claim_event(local);
            }
        }
    }
    let mut state = match probe {
        Probe::Alone => State::None,
        Probe::Record(handle) => State::Claiming(handle),
        Probe::Gated(claims) => State::Gated(claims),
    };

    let mut ref_pos = read.start;
    let mut read_off: usize = 0;

    for &(kind, len) in &read.cigar {
        let len = len as u64;
        match kind {
            CigarKind::Match => {
                let lo = ref_pos.max(win_start);
                let hi = (ref_pos + len).min(win_end);
                if lo < hi {
                    let read_lo = read_off + (lo - ref_pos) as usize;
                    let local_lo = (lo - win_start) as usize;
                    let local_hi = (hi - win_start) as usize;

                    if mode == ImportMode::Depth && matches!(state, State::None) {
                        let depth = &mut acc.fields[DEPTH][local_lo..local_hi];
                        if filter.min_bq == 0 {
                            // No quality can fall below 0, so the whole block
                            // counts without consulting one. This is the case
                            // `RecordFields::DepthOnly` leaves `read.qual`
                            // empty for, so it must not read the buffer at
                            // all.
                            for d in depth.iter_mut() {
                                *d += 1;
                            }
                        } else {
                            match read.qual.get(read_lo..read_lo + (hi - lo) as usize) {
                                Some(quals) => bump_depth(depth, quals, filter.min_bq),
                                // The quality buffer doesn't cover this block,
                                // so fall back to the per-position rule the
                                // gated path uses (`qual_at`): real score
                                // where there is one, pass where there isn't.
                                // `bump_depth` can't be handed the short slice
                                // instead -- it stops at the shorter of the
                                // two, which would drop the uncovered
                                // positions rather than pass them.
                                None => {
                                    for (i, d) in depth.iter_mut().enumerate() {
                                        if qual_at(read, read_lo + i) >= filter.min_bq {
                                            *d += 1;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for p in lo..hi {
                            let local = (p - win_start) as usize;
                            let read_i = read_off + (p - ref_pos) as usize;

                            if state.base_blocked(local) {
                                continue;
                            }

                            let gated_by_bq = qual_at(read, read_i) < filter.min_bq;
                            if !gated_by_bq {
                                acc.fields[DEPTH][local] += 1;
                            }
                            if mode == ImportMode::Composition {
                                let field = if gated_by_bq {
                                    N
                                } else {
                                    // An absent SEQ (`SEQ = *`) tallies as N,
                                    // the same bucket an unrecognized base
                                    // letter lands in.
                                    match read.seq.get(read_i).copied().unwrap_or(b'N') {
                                        b'A' => A,
                                        b'C' => C,
                                        b'G' => G,
                                        b'T' => T,
                                        _ => N,
                                    }
                                };
                                acc.fields[field][local] += 1;
                            }
                            if !gated_by_bq {
                                state.claim_base(local);
                            }
                        }
                    }
                }
                ref_pos += len;
                read_off += len as usize;
            }
            CigarKind::Del => {
                let lo = ref_pos.max(win_start);
                let hi = (ref_pos + len).min(win_end);
                // A D op takes the *base* claim only when it actually covers
                // the position, i.e. when `count_deletions` folds deletions
                // into depth. Otherwise it contributes no depth there, and
                // base-claiming would suppress the mate's real M coverage
                // whenever the two mates disagree on op class (mate1 deletes
                // where mate2 matches) -- an undercount, and the residual
                // divergence against riker/Picard-style M-only pileups. The
                // `del` event itself is claimed separately either way.
                for p in lo..hi {
                    let local = (p - win_start) as usize;
                    if filter.count_deletions && !state.base_blocked(local) {
                        acc.fields[DEPTH][local] += 1;
                        state.claim_base(local);
                    }
                    if mode == ImportMode::Composition && !state.event_blocked(local) {
                        acc.fields[DEL][local] += 1;
                        state.claim_event(local);
                    }
                }
                ref_pos += len;
            }
            CigarKind::RefSkip => {
                // Composition-only field, gated by the event claim: two mates
                // sharing a splice (N) span over the same genomic position
                // must record it once. It contributes no depth, so it never
                // takes the base claim.
                if mode == ImportMode::Composition {
                    let lo = ref_pos.max(win_start);
                    let hi = (ref_pos + len).min(win_end);
                    for p in lo..hi {
                        let local = (p - win_start) as usize;
                        if state.event_blocked(local) {
                            continue;
                        }
                        acc.fields[REF_SKIP][local] += 1;
                        state.claim_event(local);
                    }
                }
                ref_pos += len;
            }
            CigarKind::Ins => {
                // Anchor the insertion at the last reference position the
                // read consumed *before* the gap, so the anchor always falls
                // inside the read's own reference span. Anchoring at
                // `ref_pos` (the base after the gap) puts a trailing `I` one
                // past that span, where the chunk-sized query that owns the
                // anchor position never returns this read -- the insertion
                // would appear or vanish with the chunk grid. A leading `I`
                // has no preceding reference position, so it keeps `ref_pos`.
                let anchor = if ref_pos > read.start {
                    ref_pos - 1
                } else {
                    ref_pos
                };
                // Same event-claim treatment as RefSkip: an insertion
                // anchored at the same position for both mates (a shared
                // artifact in the overlap) counts once.
                if mode == ImportMode::Composition && anchor >= win_start && anchor < win_end {
                    let local = (anchor - win_start) as usize;
                    if !state.event_blocked(local) {
                        acc.fields[INS][local] += 1;
                        state.claim_event(local);
                    }
                }
                read_off += len as usize;
            }
            CigarKind::SoftClip => {
                read_off += len as usize;
            }
            CigarKind::Other => {}
        }
    }

    if let State::Claiming(handle) = state {
        mates.commit(handle);
    }
}
