//! Shared BAM/CRAM fixtures, committed under `fixtures/bam/`. The full
//! record tables and regeneration instructions live in that directory's
//! README.

use std::path::{Path, PathBuf};

// Shared across test binaries; not every consumer needs every field (e.g.
// tests/bam_walk.rs only touches `bam`).
#[allow(dead_code)]
pub struct Fixture {
    pub bam: PathBuf,
    pub cram: PathBuf,
    pub fasta: PathBuf,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bam")
}

/// The main BAM + CRAM pair (`ref1` 100bp, `ref2` 60bp; every read on
/// `ref1`). Records on `ref1` (0-based half-open), post default filter
/// (`exclude_flags = 1796` drops r4's DUP flag):
///   r1a  flag 99   mapq 60  pos 10  20M          -> [10,30), mate at 25 (paired w/ r1b)
///   r1b  flag 147  mapq 60  pos 25  20M          -> [25,45), mate at 10 (paired w/ r1a)
///   r2   flag 0    mapq 60  pos 40  10M5D10M     -> M[40,50) D[50,55) M[55,65)
///   r3   flag 0    mapq 60  pos 70  5M10N5M      -> M[70,75) N[75,85) M[85,90)
///   r4   flag 1024 mapq 60  pos 10  20M          -> excluded (DUP bit set)
///   r5   flag 0    mapq 5   pos 10  20M          -> [10,30)
///   r6   flag 0    mapq 60  pos 50  5M1I5M       -> M[50,55) Ins@54 M[55,60)
///   r7   flag 0    mapq 60  pos 12  10M          -> [12,22), BQ5 at read offset 3 (ref pos 15)
///   r8   flag 0    mapq 60  pos 90  8M, QUAL=*   -> [90,98), every base passes any min_bq
///
/// r1a/r1b overlap on [25,30); r1a starts first (coordinate-sorted fetch
/// order), so it parks and claims [10,30), and r1b's walk skips [25,30)
/// under the default `overlap = OverlapMode::ProperOnly` filter (r1a/r1b are
/// flags 99/147, i.e. proper-pair, so `ProperOnly` behaves identically to
/// `All` for this fixture).
pub fn fixture() -> Fixture {
    let dir = fixture_dir();
    Fixture {
        bam: dir.join("reads.bam"),
        cram: dir.join("reads.cram"),
        fasta: dir.join("ref.fa"),
    }
}

/// Depth at every position [0,100) on `ref1` with the default filter,
/// hand-computed from the fixture (see [`fixture`]'s record table for the
/// per-record derivation). Shared by `bam_walk.rs` and `import_bam.rs` so
/// both check against the same ground truth.
#[allow(dead_code)]
pub fn expected_default() -> [i32; 100] {
    let mut d = [0i32; 100];
    let mut add = |lo: usize, hi: usize| {
        for v in &mut d[lo..hi] {
            *v += 1;
        }
    };
    add(10, 30); // r1a (parks, walks in full)
    add(30, 45); // r1b, [25,30) already claimed by r1a
    add(40, 50); // r2 M
    // r2 D [50,55) contributes nothing to depth by default (count_deletions
    // = false).
    add(55, 65); // r2 M
    add(70, 75); // r3 M
    add(85, 90); // r3 M (N[75,85) contributes nothing)
    add(10, 30); // r5 (min_mapq default 0, so included)
    add(50, 55); // r6 M
    add(55, 60); // r6 M (Ins@55 contributes nothing to depth)
    add(12, 22); // r7
    add(90, 98); // r8 (QUAL = *, so every base clears the BQ gate)
    d
}

/// A standalone BAM (contig `dref`) holding one overlapping pair where the
/// first mate carries a D run inside the overlap and the second mate matches
/// across it. Kept out of [`fixture`] because that fixture's `ref2` is
/// asserted empty by `bam_backend.rs` / `import_bam.rs` and its `ref1` depth
/// vector is pinned by [`expected_default`].
#[allow(dead_code)]
pub fn del_overlap_bam() -> PathBuf {
    fixture_dir().join("dref.bam")
}

/// A standalone BAM (contig `pref`) holding one overlapping pair whose
/// PROPER_PAIR bit (0x2) is unset, i.e. a discordant pair. Used to pin
/// `OverlapMode`: `ProperOnly` (mosdepth match) must NOT dedup this pair,
/// `All` (riker match) must dedup it, `None` must not dedup it either.
#[allow(dead_code)]
pub fn improper_pair_overlap_bam() -> PathBuf {
    fixture_dir().join("pref.bam")
}

/// A standalone BAM (contig `mref`) holding three proper pairs whose overlap
/// spans disagree on CIGAR op class between the mates -- the exact shape
/// that a single op-class-agnostic claim bitmap gets wrong.
#[allow(dead_code)]
pub fn mixed_overlap_bam() -> PathBuf {
    fixture_dir().join("mref.bam")
}

/// A standalone BAM + CRAM pair (contig `nref`) holding two overlapping
/// proper "pairs" whose records are all unnamed (`QNAME = *`). Unnamed
/// records can't be paired with anything, so all four must count
/// independently -- on both backends, which spell an absent name differently
/// on the wire.
#[allow(dead_code)]
pub fn nameless_overlap_pair() -> (PathBuf, PathBuf, PathBuf) {
    let dir = fixture_dir();
    (
        dir.join("nref.bam"),
        dir.join("nref.cram"),
        dir.join("nref.fa"),
    )
}

/// A standalone BAM (contig `iref`) pinning insertion anchoring: a trailing
/// insertion must anchor inside the read's own reference span so that a
/// chunked import (whose per-chunk query only returns reads overlapping that
/// chunk) still sees it.
#[allow(dead_code)]
pub fn ins_anchor_bam() -> PathBuf {
    fixture_dir().join("iref.bam")
}
