//! Mate-overlap bookkeeping: coordinate-sorted `Backend::fetch` visits one
//! mate of an overlapping pair before the other, so double-counting the
//! shared span requires remembering which positions the first mate already
//! claimed and gating the second mate's increments there.
//!
//! Reads arrive coordinate-sorted, so whichever mate starts first is always
//! probed first; it "parks" its claims keyed by qname, walks normally, and
//! records every position it increments. The second mate probes into the same
//! qname, gets the claims back, and skips already-claimed positions. Claims
//! grow lazily (plain `BitVec`s, not preallocated to the window): the
//! overlap between two mates is typically a small fraction of either read,
//! so most claims never need more than a few set bits.
//!
//! [`Claims`] carries one bitmap per contribution class (see its docs): a
//! position claimed for a deletion event must not gate the mate's Match
//! coverage at that same position.
//!
//! A parked entry whose mate never arrives is dropped once the coordinate
//! sweep passes the mate start the entry was parked for, so a window full of
//! half-present pairs can't grow the map without bound.

use bitvec::boxed::BitBox;
use bitvec::vec::BitVec;
use rustc_hash::FxHashMap;

use super::OverlapMode;
use super::record::{AlignedRead, CigarKind};

/// Mate-unmapped flag bit (0x8). CRAM-decoded records can set
/// `mate_same_contig`/`mate_start` even when the mate is unmapped (it gets
/// placed at the read's own position); checking this bit directly avoids
/// treating that placeholder as a real overlapping mate.
const MATE_UNMAPPED: u16 = 0x8;
/// Proper-pair flag (0x2). Gates parking/consuming under
/// [`OverlapMode::ProperOnly`], mirroring mosdepth's
/// `rec.flag.proper_pair` gate on its own overlap correction.
const PROPER_PAIR: u16 = 0x2;
/// Secondary alignment (0x100) and supplementary alignment (0x800). Both are
/// extra records for a molecule that already has (or will have) a primary
/// record under the same qname; the default `exclude_flags` (1796) drops
/// secondary but not supplementary, and either can be reintroduced by a
/// caller-supplied filter. Neither may enter the pairing state machine: a
/// same-qname secondary/supplementary record must never park (it isn't the
/// "real" first mate) and must never consume a pending claim (it isn't the
/// "real" second mate either), or it can silently steal / duplicate another
/// pair's overlap accounting.
const SECONDARY: u16 = 0x100;
const SUPPLEMENTARY: u16 = 0x800;

/// How many parked entries the map must hold before an unmatched-entry purge
/// is worth its linear scan. Purging is amortized: the map only grows on a
/// park, so a scan every time the threshold is exceeded is O(1) per park.
const PURGE_THRESHOLD: usize = 64;

pub enum Probe {
    /// No overlapping mate in play; walk and increment normally.
    Alone,
    /// This read starts at or before its mate and the two spans overlap:
    /// walk normally, but record every incremented position on `ClaimHandle`
    /// so the mate can be gated.
    Record(ClaimHandle),
    /// The mate already walked and claimed some positions; skip increments
    /// where the matching class bit is set.
    Gated(Claims),
}

/// One parked read's claimed positions, split by contribution class.
///
/// A single op-class-agnostic bitmap conflates "this pair already covered
/// this position" with "this pair already recorded an event here". Mates
/// routinely disagree on op class inside their overlap (one deletes or
/// splices where the other matches), and the coarse bitmap then lets the
/// first mate's event suppress the second mate's real coverage: depth and the
/// per-base tallies silently vanish in Composition mode.
///
/// `base` holds positions where the pair already contributed depth (a
/// BQ-passing Match base, or a Del position when `count_deletions` makes
/// deletions cover); it gates depth and the a/c/g/t/n tallies. `event` holds
/// positions where the pair already recorded a del/ref_skip/ins event; it
/// gates only those three fields.
pub struct Claims {
    base: BitBox,
    event: BitBox,
}

impl Claims {
    pub fn base_claimed(&self, window_idx: usize) -> bool {
        is_set(&self.base, window_idx)
    }

    pub fn event_claimed(&self, window_idx: usize) -> bool {
        is_set(&self.event, window_idx)
    }
}

fn is_set(bits: &BitBox, idx: usize) -> bool {
    bits.get(idx).is_some_and(|b| *b)
}

/// Accumulates claimed positions for one parked read. Not tied back to
/// `MateBuffer` by a lifetime borrow; the caller explicitly hands it back
/// via [`MateBuffer::commit`] once the walk over the record finishes.
pub struct ClaimHandle {
    qname: Vec<u8>,
    /// Where the mate is expected, so an entry whose mate never arrives can
    /// be dropped once the coordinate sweep passes it.
    expected_mate_start: u64,
    base: BitVec,
    event: BitVec,
}

impl ClaimHandle {
    /// Mark `window_idx` (a position relative to `win_start`) as covered by
    /// this pair. Gates the mate's depth and base tallies there.
    pub fn claim_base(&mut self, window_idx: usize) {
        set_growing(&mut self.base, window_idx);
    }

    /// Mark `window_idx` as already carrying a del/ref_skip/ins event for
    /// this pair. Gates only the mate's event fields there.
    pub fn claim_event(&mut self, window_idx: usize) {
        set_growing(&mut self.event, window_idx);
    }
}

/// Claimed positions are sparse relative to the window, so the bitmaps grow
/// on demand instead of being preallocated to the window length.
fn set_growing(bits: &mut BitVec, window_idx: usize) {
    if window_idx >= bits.len() {
        bits.resize(window_idx + 1, false);
    }
    bits.set(window_idx, true);
}

struct Parked {
    claims: Claims,
    expected_mate_start: u64,
}

#[derive(Default)]
pub struct MateBuffer {
    pending: FxHashMap<Vec<u8>, Parked>,
}

impl MateBuffer {
    pub fn probe(&mut self, read: &AlignedRead, mode: OverlapMode) -> Probe {
        // An unnamed record (SAM `QNAME = *`) can't be paired with anything:
        // every such record hashes to the same key, so pairing them would let
        // unrelated reads consume each other's claims. Treat them as unpaired.
        if read.qname.is_empty() {
            return Probe::Alone;
        }
        // A secondary/supplementary record is never a pairing participant:
        // neither eligible to consume a pending claim (it isn't the mate
        // that left it) nor to park one (it isn't the molecule's primary
        // alignment). Checked before the pending-map lookup so it can't
        // steal a stale entry left behind by an excluded primary mate.
        if !plausible_mate(read, mode) {
            return Probe::Alone;
        }
        if let Some(parked) = self.pending.remove(&read.qname) {
            return Probe::Gated(parked.claims);
        }
        if should_park(read) {
            // Reads arrive coordinate-sorted, so any entry still waiting for a
            // mate that would have started before this read is never going to
            // be claimed.
            if self.pending.len() > PURGE_THRESHOLD {
                self.pending
                    .retain(|_, parked| parked.expected_mate_start >= read.start);
            }
            return Probe::Record(ClaimHandle {
                qname: read.qname.clone(),
                expected_mate_start: read.mate_start.max(0) as u64,
                base: BitVec::new(),
                event: BitVec::new(),
            });
        }
        Probe::Alone
    }

    /// Finalize a parked read's claims into the pending map, to be picked up
    /// by the mate's later `probe` call.
    pub fn commit(&mut self, handle: ClaimHandle) {
        self.pending.insert(
            handle.qname,
            Parked {
                claims: Claims {
                    base: handle.base.into_boxed_bitslice(),
                    event: handle.event.into_boxed_bitslice(),
                },
                expected_mate_start: handle.expected_mate_start,
            },
        );
    }

    /// How many parked entries are still waiting for their mate. Test-facing.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Drop any unmatched claims (mate absent from this window/region).
    /// Called between independent walk windows so stale entries never leak
    /// across them.
    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

/// Necessary conditions for `read` to plausibly be one half of a mate pair
/// in the overlap sense: primary, paired, pointing at a mapped mate on the
/// same contig, and (under [`OverlapMode::ProperOnly`]) flagged proper-pair.
/// Applied uniformly to both sides of pairing (parking a claim and consuming
/// one) so a probe that finds a pending entry can't assume "same qname"
/// alone means "this is the mate" (see the module doc and the
/// `SECONDARY`/`SUPPLEMENTARY` comment above).
fn plausible_mate(read: &AlignedRead, mode: OverlapMode) -> bool {
    if mode == OverlapMode::ProperOnly && read.flags & PROPER_PAIR == 0 {
        return false;
    }
    read.paired
        && read.mate_same_contig
        && read.flags & (SECONDARY | SUPPLEMENTARY | MATE_UNMAPPED) == 0
}

fn should_park(read: &AlignedRead) -> bool {
    read.mate_start >= read.start as i64
        && (read.mate_start as u64) < read.start + read_ref_span(read)
}

/// Sum of reference-consuming CIGAR op lengths (M/D/N), i.e. how far this
/// read's alignment extends along the reference.
fn read_ref_span(read: &AlignedRead) -> u64 {
    read.cigar
        .iter()
        .filter(|(kind, _)| matches!(kind, CigarKind::Match | CigarKind::Del | CigarKind::RefSkip))
        .map(|(_, len)| *len as u64)
        .sum()
}
