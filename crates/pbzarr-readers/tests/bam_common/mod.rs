//! Shared BAM/CRAM test fixture: a hand-written SAM covering paired reads,
//! deletions, ref-skips, a duplicate, a low-mapq read, an insertion, and a
//! per-base quality outlier, synthesized via the system `samtools` (skipped
//! if unavailable, mirroring the `d4tools` pattern in `tests/import_d4.rs`).

use std::path::{Path, PathBuf};
use std::process::Command;

// Shared across test binaries; not every consumer needs every field (e.g.
// tests/bam_walk.rs only touches `bam`).
#[allow(dead_code)]
pub struct Fixture {
    pub bam: PathBuf,
    pub cram: PathBuf,
    pub fasta: PathBuf,
}

fn have_samtools() -> bool {
    Command::new("samtools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `ref1` is 100bp, `ref2` is 60bp, both a repeating `ACGT` motif so the
/// content is deterministic but not aligned with any read's own bases.
fn write_fasta(path: &Path) {
    let motif = "ACGT";
    let ref1: String = motif.chars().cycle().take(100).collect();
    let ref2: String = motif.chars().cycle().take(60).collect();
    let mut out = String::new();
    out.push_str(">ref1\n");
    for chunk in ref1.as_bytes().chunks(60) {
        out.push_str(std::str::from_utf8(chunk).unwrap());
        out.push('\n');
    }
    out.push_str(">ref2\n");
    for chunk in ref2.as_bytes().chunks(60) {
        out.push_str(std::str::from_utf8(chunk).unwrap());
        out.push('\n');
    }
    std::fs::write(path, out).unwrap();
}

/// Fixture records on `ref1` (0-based half-open), post default filter
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
///
/// One SAM data line. `qual` defaults to all-`?` (Phred 30) unless
/// overridden per-record (only `r7` needs a per-base outlier).
struct SamRecord {
    qname: &'static str,
    flag: u16,
    rname: &'static str,
    pos_0based: u64,
    mapq: u8,
    cigar: &'static str,
    rnext: &'static str,
    pnext_0based: i64,
    seq: String,
    qual: String,
}

fn base_run(base: char, len: usize) -> String {
    std::iter::repeat_n(base, len).collect()
}

fn qual_run(len: usize) -> String {
    base_run('?', len) // Phred 30
}

fn write_sam(path: &Path) {
    let records = vec![
        SamRecord {
            qname: "r1",
            flag: 99,
            rname: "ref1",
            pos_0based: 10,
            mapq: 60,
            cigar: "20M",
            rnext: "=",
            pnext_0based: 25,
            seq: base_run('A', 20),
            qual: qual_run(20),
        },
        SamRecord {
            qname: "r1",
            flag: 147,
            rname: "ref1",
            pos_0based: 25,
            mapq: 60,
            cigar: "20M",
            rnext: "=",
            pnext_0based: 10,
            seq: base_run('A', 20),
            qual: qual_run(20),
        },
        SamRecord {
            qname: "r2",
            flag: 0,
            rname: "ref1",
            pos_0based: 40,
            mapq: 60,
            cigar: "10M5D10M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('A', 20),
            qual: qual_run(20),
        },
        SamRecord {
            qname: "r3",
            flag: 0,
            rname: "ref1",
            pos_0based: 70,
            mapq: 60,
            cigar: "5M10N5M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('A', 10),
            qual: qual_run(10),
        },
        SamRecord {
            qname: "r4",
            flag: 1024,
            rname: "ref1",
            pos_0based: 10,
            mapq: 60,
            cigar: "20M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('A', 20),
            qual: qual_run(20),
        },
        SamRecord {
            qname: "r5",
            flag: 0,
            rname: "ref1",
            pos_0based: 10,
            mapq: 5,
            cigar: "20M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('A', 20),
            qual: qual_run(20),
        },
        SamRecord {
            qname: "r6",
            flag: 0,
            rname: "ref1",
            pos_0based: 50,
            mapq: 60,
            cigar: "5M1I5M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('C', 11),
            qual: qual_run(11),
        },
        SamRecord {
            qname: "r7",
            flag: 0,
            rname: "ref1",
            pos_0based: 12,
            mapq: 60,
            cigar: "10M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('G', 10),
            qual: {
                // Read offset 3 gets Phred 5 ('&'); the rest stay Phred 30 ('?').
                let mut q: Vec<u8> = qual_run(10).into_bytes();
                q[3] = b'&';
                String::from_utf8(q).unwrap()
            },
        },
        SamRecord {
            qname: "r8",
            flag: 0,
            rname: "ref1",
            pos_0based: 90,
            mapq: 60,
            cigar: "8M",
            rnext: "*",
            pnext_0based: -1,
            seq: base_run('T', 8),
            // QUAL = *: legal, and means "quality unavailable", so every base
            // clears any `min_bq`.
            qual: "*".to_string(),
        },
    ];

    let mut out = String::new();
    out.push_str("@HD\tVN:1.6\tSO:unsorted\n");
    out.push_str("@SQ\tSN:ref1\tLN:100\n");
    out.push_str("@SQ\tSN:ref2\tLN:60\n");
    for r in &records {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t0\t{}\t{}\n",
            r.qname,
            r.flag,
            r.rname,
            r.pos_0based + 1,
            r.mapq,
            r.cigar,
            r.rnext,
            r.pnext_0based + 1,
            r.seq,
            r.qual,
        ));
    }
    std::fs::write(path, out).unwrap();
}

/// Depth at every position [0,100) on `ref1` with the default filter,
/// hand-computed from the fixture (see `tests/bam_walk.rs` for the per-record
/// derivation). Shared by `bam_walk.rs` and `import_bam.rs` so both check
/// against the same ground truth.
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

/// A standalone BAM (own contig `dref`, own temp files) holding one
/// overlapping pair where the first mate carries a D run inside the overlap
/// and the second mate matches across it:
///   r8a flag 99  pos 10  5M5D10M -> M[10,15) D[15,20) M[20,30), mate at 15
///   r8b flag 147 pos 15  20M     -> M[15,35),             mate at 10
///
/// Kept out of [`fixture`] because that fixture's `ref2` is asserted empty by
/// `bam_backend.rs` / `import_bam.rs` and its `ref1` depth vector is pinned by
/// `expected_default`.
#[allow(dead_code)]
pub fn del_overlap_bam(dir: &Path) -> Option<PathBuf> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let records = vec![
        format!(
            "r8\t99\tdref\t11\t60\t5M5D10M\t=\t16\t0\t{}\t{}\n",
            base_run('A', 15),
            qual_run(15)
        ),
        format!(
            "r8\t147\tdref\t16\t60\t20M\t=\t11\t0\t{}\t{}\n",
            base_run('A', 20),
            qual_run(20)
        ),
    ];
    Some(build_bam(dir, "dref", "dref", 60, &records))
}

/// A standalone BAM (own contig `pref`, own temp files) holding one
/// overlapping pair whose PROPER_PAIR bit (0x2) is unset, i.e. a discordant
/// pair: same flags as `fixture`'s r1a/r1b (99/147) with bit 0x2 cleared
/// (97/145).
///   r9a flag 97  pos 10  20M -> [10,30), mate at 25, NOT proper-pair
///   r9b flag 145 pos 25  20M -> [25,45), mate at 10, NOT proper-pair
///
/// Overlap span [25,30). Used to pin `OverlapMode`: `ProperOnly` (mosdepth
/// match) must NOT dedup this pair (depth 2 across the overlap), `All`
/// (riker match) must dedup it (depth 1), `None` must not dedup it either
/// (depth 2).
#[allow(dead_code)]
pub fn improper_pair_overlap_bam(dir: &Path) -> Option<PathBuf> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let records = vec![
        format!(
            "r9\t97\tpref\t11\t60\t20M\t=\t26\t0\t{}\t{}\n",
            base_run('A', 20),
            qual_run(20)
        ),
        format!(
            "r9\t145\tpref\t26\t60\t20M\t=\t11\t0\t{}\t{}\n",
            base_run('A', 20),
            qual_run(20)
        ),
    ];
    Some(build_bam(dir, "pref", "pref", 60, &records))
}

/// Sort + index `records` (already-formatted SAM data lines) into
/// `<dir>/<stem>.bam` over a single contig.
#[allow(dead_code)]
fn build_bam(dir: &Path, stem: &str, contig: &str, contig_len: u64, records: &[String]) -> PathBuf {
    let sam = dir.join(format!("{stem}.sam"));
    let mut out = format!("@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:{contig}\tLN:{contig_len}\n");
    for r in records {
        out.push_str(r);
    }
    std::fs::write(&sam, out).unwrap();

    let bam = dir.join(format!("{stem}.bam"));
    assert!(
        Command::new("samtools")
            .args(["sort", "-o"])
            .arg(&bam)
            .arg(&sam)
            .status()
            .unwrap()
            .success(),
        "samtools sort failed"
    );
    assert!(
        Command::new("samtools")
            .args(["index"])
            .arg(&bam)
            .status()
            .unwrap()
            .success(),
        "samtools index failed"
    );
    bam
}

/// A standalone BAM (contig `mref`, 120bp) holding three proper pairs whose
/// overlap spans disagree on CIGAR op class between the mates -- the exact
/// shape that a single op-class-agnostic claim bitmap gets wrong:
///   m1a 99  pos 10  5M5D10M -> D[15,20) against m1b's M
///   m1b 147 pos 15  20M
///   m2a 99  pos 40  5M5N10M -> N[45,50) against m2b's M
///   m2b 147 pos 45  20M
///   m3a 99  pos 70  5M2I5M  -> insertion anchor against m3b's M
///   m3b 147 pos 74  10M
#[allow(dead_code)]
pub fn mixed_overlap_bam(dir: &Path) -> Option<PathBuf> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let line = |qname: &str, flag: u16, pos1: u64, cigar: &str, pnext1: u64, seq_len: usize| {
        format!(
            "{qname}\t{flag}\tmref\t{pos1}\t60\t{cigar}\t=\t{pnext1}\t0\t{}\t{}\n",
            base_run('A', seq_len),
            qual_run(seq_len)
        )
    };
    let records = vec![
        line("m1", 99, 11, "5M5D10M", 16, 15),
        line("m1", 147, 16, "20M", 11, 20),
        line("m2", 99, 41, "5M5N10M", 46, 15),
        line("m2", 147, 46, "20M", 41, 20),
        line("m3", 99, 71, "5M2I5M", 75, 12),
        line("m3", 147, 75, "10M", 71, 10),
    ];
    Some(build_bam(dir, "mref", "mref", 120, &records))
}

/// A standalone BAM + CRAM pair (contig `nref`, 60bp) holding two overlapping
/// proper "pairs" whose records are all unnamed (`QNAME = *`):
///   pos 10 20M flag 99  (x2), mate at 25
///   pos 25 20M flag 147 (x2), mate at 10
/// Unnamed records can't be paired with anything, so all four must count
/// independently -- on both backends, which spell an absent name differently
/// on the wire.
#[allow(dead_code)]
pub fn nameless_overlap_pair(dir: &Path) -> Option<(PathBuf, PathBuf, PathBuf)> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let fasta = dir.join("nref.fa");
    let seq: String = "ACGT".chars().cycle().take(60).collect();
    std::fs::write(&fasta, format!(">nref\n{seq}\n")).unwrap();

    let line = |flag: u16, pos1: u64, pnext1: u64| {
        format!(
            "*\t{flag}\tnref\t{pos1}\t60\t20M\t=\t{pnext1}\t0\t{}\t{}\n",
            base_run('A', 20),
            qual_run(20)
        )
    };
    let records = vec![
        line(99, 11, 26),
        line(99, 11, 26),
        line(147, 26, 11),
        line(147, 26, 11),
    ];
    let bam = build_bam(dir, "nref", "nref", 60, &records);

    let cram = dir.join("nref.cram");
    assert!(
        Command::new("samtools")
            .args(["view", "-C", "-T"])
            .arg(&fasta)
            .arg("-o")
            .arg(&cram)
            .arg(&bam)
            .status()
            .unwrap()
            .success(),
        "samtools view -C failed"
    );
    assert!(
        Command::new("samtools")
            .args(["index"])
            .arg(&cram)
            .status()
            .unwrap()
            .success(),
        "samtools index (cram) failed"
    );

    Some((bam, cram, fasta))
}

/// A standalone BAM (contig `iref`, 40bp) pinning insertion anchoring:
///   r10 pos 10 5M1I -> trailing insertion, ref span [10,15)
///   r11 pos 20 1I5M -> leading insertion, ref span [20,25)
/// The trailing insertion must anchor inside the read's own reference span so
/// that a chunked import (whose per-chunk query only returns reads
/// overlapping that chunk) still sees it.
#[allow(dead_code)]
pub fn ins_anchor_bam(dir: &Path) -> Option<PathBuf> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let records = vec![
        format!(
            "r10\t0\tiref\t11\t60\t5M1I\t*\t0\t0\t{}\t{}\n",
            base_run('A', 6),
            qual_run(6)
        ),
        format!(
            "r11\t0\tiref\t21\t60\t1I5M\t*\t0\t0\t{}\t{}\n",
            base_run('A', 6),
            qual_run(6)
        ),
    ];
    Some(build_bam(dir, "iref", "iref", 40, &records))
}

/// Build the BAM + CRAM fixture pair under `dir`. Returns `None` when
/// `samtools` isn't on `PATH`, so callers self-skip.
pub fn fixture(dir: &Path) -> Option<Fixture> {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return None;
    }

    let fasta = dir.join("ref.fa");
    write_fasta(&fasta);

    let sam = dir.join("in.sam");
    write_sam(&sam);

    let bam = dir.join("reads.bam");
    let status = Command::new("samtools")
        .args(["sort", "-o"])
        .arg(&bam)
        .arg(&sam)
        .status()
        .unwrap();
    assert!(status.success(), "samtools sort failed");

    let status = Command::new("samtools")
        .args(["index"])
        .arg(&bam)
        .status()
        .unwrap();
    assert!(status.success(), "samtools index (bam) failed");

    let cram = dir.join("reads.cram");
    let status = Command::new("samtools")
        .args(["view", "-C", "-T"])
        .arg(&fasta)
        .arg("-o")
        .arg(&cram)
        .arg(&bam)
        .status()
        .unwrap();
    assert!(status.success(), "samtools view -C failed");

    let status = Command::new("samtools")
        .args(["index"])
        .arg(&cram)
        .status()
        .unwrap();
    assert!(status.success(), "samtools index (cram) failed");

    Some(Fixture { bam, cram, fasta })
}
