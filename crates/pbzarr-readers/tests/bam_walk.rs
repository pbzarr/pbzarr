//! Depth-walk expectations, hand-computed from the `bam_common` fixture. See
//! `bam_common::write_sam`'s doc comment for the per-record table this file's
//! expectations are derived from.

mod bam_common;

use pbzarr_readers::bam::backend::Backend;
use pbzarr_readers::bam::mate::{MateBuffer, Probe};
use pbzarr_readers::bam::walk::{Accumulators, FIELDS, ImportMode, walk_record};
use pbzarr_readers::bam::{AlignedRead, CigarKind, DepthFilter, OverlapMode};

// Hand-computed depth ground truth (r1a+r1b mate overlap counted once, r2
// M/D/M, r3 M with N skipped, r5, r6 M/Ins-skipped/M, r7, r8) lives in
// `bam_common::expected_default`, shared with `tests/import_bam.rs`.
use bam_common::expected_default;

fn walk_window(bam: &std::path::Path, filter: &DepthFilter) -> Vec<i32> {
    walk_sub_window(bam, filter, 0, 100)
}

/// Walk `[win_start, win_end)` only, fetching from the same bounds (so
/// records entirely outside the window are never even decoded, matching how
/// `run_pipeline` will drive this per-chunk in the real import path).
fn walk_sub_window(
    bam: &std::path::Path,
    filter: &DepthFilter,
    win_start: u64,
    win_end: u64,
) -> Vec<i32> {
    let (mut backend, _genome) = Backend::open(bam, None).unwrap();
    let window_len = (win_end - win_start) as usize;
    let mut acc = Accumulators::new(1, window_len);
    let mut mates = MateBuffer::default();
    backend
        .fetch("ref1", win_start, win_end, &mut |read| {
            walk_record(
                read,
                win_start,
                win_end,
                filter,
                ImportMode::Depth,
                &mut acc,
                &mut mates,
            );
            Ok(())
        })
        .unwrap();
    acc.fields[0].clone()
}

/// Each case flips exactly one `DepthFilter` knob off its default and
/// mutates the hand-computed default-filter depth vector by the effect that
/// single knob is expected to have; the rest of the fixture is untouched.
#[test]
fn depth_single_knob_filter_permutations() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };

    struct Case {
        name: &'static str,
        filter: DepthFilter,
        mutate: fn(&mut [i32; 100]),
    }
    let cases = [
        Case {
            name: "default",
            filter: DepthFilter::default(),
            mutate: |_| {},
        },
        Case {
            name: "overlap_none",
            filter: DepthFilter {
                overlap: OverlapMode::None,
                ..DepthFilter::default()
            },
            // r1b now also counts its own [25,30), on top of r1a's.
            mutate: |e| {
                for v in &mut e[25..30] {
                    *v += 1;
                }
            },
        },
        Case {
            name: "min_mapq_10",
            filter: DepthFilter {
                min_mapq: 10,
                ..DepthFilter::default()
            },
            // r5 (mapq 5) drops out entirely: subtract its [10,30) contribution.
            mutate: |e| {
                for v in &mut e[10..30] {
                    *v -= 1;
                }
            },
        },
        Case {
            name: "min_bq_20",
            filter: DepthFilter {
                min_bq: 20,
                ..DepthFilter::default()
            },
            // r7's read offset 3 (ref pos 12+3=15) has BQ 5, below the 20 gate.
            mutate: |e| e[15] -= 1,
        },
        Case {
            name: "count_deletions_true",
            filter: DepthFilter {
                count_deletions: true,
                ..DepthFilter::default()
            },
            // r2's D-run [50,55) (10M5D10M at pos 40) counts toward depth when
            // `count_deletions` is opted in; everything else is unchanged.
            mutate: |e| {
                for v in &mut e[50..55] {
                    *v += 1;
                }
            },
        },
    ];

    for case in cases {
        let depth = walk_window(&fx.bam, &case.filter);
        let mut expected = expected_default();
        (case.mutate)(&mut expected);
        assert_eq!(depth, expected.to_vec(), "case={}", case.name);
    }
}

/// A D run in the first mate must not claim the overlap slot when deletions
/// don't count toward depth: r8a's D[15,20) contributes nothing there, so
/// r8b's M across the same span is the only coverage and must survive the
/// overlap gate. Depth over [10,35) is 1 either way -- with
/// `count_deletions` the D itself supplies the base and gates r8b, without it
/// r8b supplies it -- so the same expectation pins both filter settings.
#[test]
fn del_run_does_not_claim_overlap_slot_when_deletions_uncounted() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(bam) = bam_common::del_overlap_bam(dir.path()) else {
        return;
    };

    let mut expected = vec![0i32; 60];
    for v in &mut expected[10..35] {
        *v = 1;
    }

    for count_deletions in [true, false] {
        let filter = DepthFilter {
            count_deletions,
            ..DepthFilter::default()
        };
        let (mut backend, _genome) = Backend::open(&bam, None).unwrap();
        let mut acc = Accumulators::new(1, 60);
        let mut mates = MateBuffer::default();
        backend
            .fetch("dref", 0, 60, &mut |read| {
                walk_record(
                    read,
                    0,
                    60,
                    &filter,
                    ImportMode::Depth,
                    &mut acc,
                    &mut mates,
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(acc.fields[0], expected, "count_deletions={count_deletions}");
    }
}

/// Pins `OverlapMode` against a discordant (PROPER_PAIR unset) overlapping
/// pair (`bam_common::improper_pair_overlap_bam`, flags 97/145, overlap
/// [25,30)): `ProperOnly` (mosdepth match) double-counts the overlap since
/// neither mate is proper-pair-flagged; `All` (riker match) dedups it to
/// one count regardless of the flag; `None` double-counts it too, same as
/// `ProperOnly` here but for a different reason (dedup is off entirely).
#[test]
fn overlap_mode_gates_on_proper_pair_flag() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(bam) = bam_common::improper_pair_overlap_bam(dir.path()) else {
        return;
    };

    let mut expected_deduped = vec![0i32; 60];
    for v in &mut expected_deduped[10..45] {
        *v = 1;
    }
    let mut expected_double_counted = expected_deduped.clone();
    for v in &mut expected_double_counted[25..30] {
        *v += 1;
    }

    let cases = [
        (OverlapMode::ProperOnly, &expected_double_counted),
        (OverlapMode::All, &expected_deduped),
        (OverlapMode::None, &expected_double_counted),
    ];

    for (mode, expected) in cases {
        let filter = DepthFilter {
            overlap: mode,
            ..DepthFilter::default()
        };
        let (mut backend, _genome) = Backend::open(&bam, None).unwrap();
        let mut acc = Accumulators::new(1, 60);
        let mut mates = MateBuffer::default();
        backend
            .fetch("pref", 0, 60, &mut |read| {
                walk_record(
                    read,
                    0,
                    60,
                    &filter,
                    ImportMode::Depth,
                    &mut acc,
                    &mut mates,
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(acc.fields[0], *expected, "overlap={mode:?}");
    }
}

/// `r8` carries `QUAL = *` (no per-base qualities) alongside a real `8M`
/// CIGAR, which is legal SAM. Its bases must count at any `min_bq` -- absent
/// quality is unknown, not zero -- and the walk must not index past the empty
/// quality buffer. Both backends must agree, since only one of them
/// materializes 0xFF placeholder scores.
#[test]
fn qual_star_record_counts_at_every_min_bq() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };

    for min_bq in [0u8, 30, 255] {
        let filter = DepthFilter {
            min_bq,
            ..DepthFilter::default()
        };
        for (label, path, reference) in [
            ("bam", &fx.bam, None),
            ("cram", &fx.cram, Some(fx.fasta.as_path())),
        ] {
            let (mut backend, _genome) = Backend::open(path, reference).unwrap();
            let mut acc = Accumulators::new(1, 100);
            let mut mates = MateBuffer::default();
            backend
                .fetch("ref1", 0, 100, &mut |read| {
                    walk_record(
                        read,
                        0,
                        100,
                        &filter,
                        ImportMode::Depth,
                        &mut acc,
                        &mut mates,
                    );
                    Ok(())
                })
                .unwrap();
            for pos in 90..98 {
                assert_eq!(acc.fields[0][pos], 1, "{label} min_bq={min_bq} pos={pos}");
            }
        }
    }
}

/// Composition mode tallies a `QUAL = *` record's bases by letter (r8 is
/// all-T), not into `n`: the base is known even when its quality isn't.
#[test]
fn qual_star_record_tallies_bases_in_composition() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    let filter = DepthFilter {
        min_bq: 30,
        ..DepthFilter::default()
    };
    let fields = walk_fields(&fx.bam, "ref1", 100, &filter, ImportMode::Composition);
    assert!(
        fields[4][90..98].iter().all(|&v| v == 1),
        "t: {:?}",
        &fields[4][90..98]
    ); // T
    assert!(
        fields[5][90..98].iter().all(|&v| v == 0),
        "n: {:?}",
        &fields[5][90..98]
    );
}

/// Walk `[0, len)` on `contig` in `mode`, returning every accumulator field.
fn walk_fields(
    bam: &std::path::Path,
    contig: &str,
    len: u64,
    filter: &DepthFilter,
    mode: ImportMode,
) -> Vec<Vec<i32>> {
    let n_fields = mode.n_fields();
    let (mut backend, _genome) = Backend::open(bam, None).unwrap();
    let mut acc = Accumulators::new(n_fields, len as usize);
    let mut mates = MateBuffer::default();
    backend
        .fetch(contig, 0, len, &mut |read| {
            walk_record(read, 0, len, filter, mode, &mut acc, &mut mates);
            Ok(())
        })
        .unwrap();
    acc.fields.clone()
}

/// Composition mode must produce exactly the depth Depth mode produces. The
/// two modes only diverge if the overlap claim bitmap is op-class-agnostic:
/// a first mate's D (or N, or insertion anchor) would then claim a position
/// it contributed no depth at, gating the second mate's Match wholesale and
/// silently losing depth in Composition mode only. Checked across every
/// (overlap mode x count_deletions) combination on a fixture whose mates
/// disagree on op class inside each overlap.
///
/// Also pins the composition sum rule on this fixture: depth == a+c+g+t+n at
/// every position, plus del where deletions count toward depth. Two fixture
/// properties make the sum exact and neither generalizes: every qual clears
/// the default `min_bq = 0` (so no base is diverted to `n` without also
/// incrementing depth), and the D-carrying mate sorts first (so with
/// `count_deletions` its D takes the base claim and the mate's M is gated,
/// rather than the M claiming first and the mate's D adding a `del` event on
/// top of an already-counted base). The depth == depth assertion above is the
/// order-independent one.
#[test]
fn composition_depth_matches_depth_mode_across_modes() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(bam) = bam_common::mixed_overlap_bam(dir.path()) else {
        return;
    };

    const LEN: u64 = 120;
    for overlap in [OverlapMode::None, OverlapMode::ProperOnly, OverlapMode::All] {
        for count_deletions in [false, true] {
            let filter = DepthFilter {
                overlap,
                count_deletions,
                ..DepthFilter::default()
            };
            let depth_only = walk_fields(&bam, "mref", LEN, &filter, ImportMode::Depth);
            let composition = walk_fields(&bam, "mref", LEN, &filter, ImportMode::Composition);
            assert_eq!(
                depth_only[0], composition[0],
                "depth mismatch: overlap={overlap:?} count_deletions={count_deletions}"
            );

            for (pos, &depth) in composition[0].iter().enumerate() {
                let mut sum: i32 = (1..=5).map(|f| composition[f][pos]).sum();
                if count_deletions {
                    sum += composition[7][pos];
                }
                assert_eq!(
                    depth, sum,
                    "depth != base sum at {pos}: overlap={overlap:?} \
                     count_deletions={count_deletions}"
                );
            }
        }
    }
}

/// Walk window [27,45) cuts through the r1a/r1b overlap span [25,30): r1a's
/// own Match op is clipped to [27,30) (its bases at 25,26 fall outside the
/// window and are never visited or claimed), so r1a still parks and claims
/// [27,30) within this window, and r1b's walk still skips exactly that
/// clipped claimed range. r2's M[40,50) is also clipped to [40,45).
///
/// Local index 0 == abs pos 27; window length 18 == abs [27,45).
/// Expected (hand-computed from the fixture table):
///   abs 27-29: r1a + r5           -> depth 2
///   abs 30-39: r1b only           -> depth 1
///   abs 40-44: r1b + r2 (M start) -> depth 2
#[test]
fn depth_sub_window_clips_ops_and_preserves_overlap_gating() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    let depth = walk_sub_window(&fx.bam, &DepthFilter::default(), 27, 45);

    let mut expected = vec![2i32; 3]; // abs 27,28,29
    expected.extend(std::iter::repeat_n(1i32, 10)); // abs 30..39
    expected.extend(std::iter::repeat_n(2i32, 5)); // abs 40..44
    assert_eq!(depth.len(), 18);
    assert_eq!(depth, expected);
}

/// A 20M paired read at `start` whose mate starts at `mate_start`, built
/// directly rather than through a fixture so the pairing state machine can be
/// driven with record shapes samtools won't write (an empty qname, or a
/// hundred half-present pairs).
fn synthetic_read(qname: &str, flags: u16, start: u64, mate_start: i64) -> AlignedRead {
    AlignedRead {
        flags,
        mapq: 60,
        start,
        mate_start,
        mate_same_contig: true,
        paired: true,
        qname: qname.as_bytes().to_vec(),
        cigar: vec![(CigarKind::Match, 20)],
        seq: vec![b'A'; 20],
        qual: vec![30u8; 20],
    }
}

/// Unnamed records (SAM `QNAME = *`) all hash to the same key, so pairing
/// them would let unrelated reads consume each other's claims. Two nameless
/// "pairs" must therefore count as four independent reads.
#[test]
fn nameless_records_never_pair() {
    let reads = [
        synthetic_read("", 99, 10, 25),
        synthetic_read("", 99, 10, 25),
        synthetic_read("", 147, 25, 10),
        synthetic_read("", 147, 25, 10),
    ];

    let filter = DepthFilter::default();
    let mut acc = Accumulators::new(1, 60);
    let mut mates = MateBuffer::default();
    for read in &reads {
        walk_record(
            read,
            0,
            60,
            &filter,
            ImportMode::Depth,
            &mut acc,
            &mut mates,
        );
    }

    let mut expected = vec![0i32; 60];
    for v in &mut expected[10..25] {
        *v = 2;
    }
    for v in &mut expected[25..30] {
        *v = 4;
    }
    for v in &mut expected[30..45] {
        *v = 2;
    }
    assert_eq!(acc.fields[0], expected);
}

/// A parked read whose mate never arrives must not sit in the buffer for the
/// rest of the window: once the coordinate sweep passes the mate start it was
/// parked for, the entry is unclaimable and gets purged. The purge is
/// amortized, so it only fires once the map is large enough to be worth the
/// scan.
#[test]
fn parked_entries_drain_once_the_sweep_passes_their_mate() {
    let mut mates = MateBuffer::default();
    // All parked at the same start, so none of them is stale relative to the
    // others and the buffer really does grow past the purge threshold.
    let n = 70;
    for i in 0..n {
        let read = synthetic_read(&format!("q{i}"), 99, 0, 5);
        let Probe::Record(handle) = mates.probe(&read, OverlapMode::ProperOnly) else {
            panic!("read {i} should park");
        };
        mates.commit(handle);
    }
    assert_eq!(mates.pending_len(), n);

    // Every parked mate was expected well before 1000.
    let late = synthetic_read("late", 99, 1000, 1005);
    let Probe::Record(handle) = mates.probe(&late, OverlapMode::ProperOnly) else {
        panic!("late read should park");
    };
    assert_eq!(mates.pending_len(), 0, "stale entries should be purged");
    mates.commit(handle);
    assert_eq!(mates.pending_len(), 1);
}

/// The `QNAME = *` rule has to hold on both backends, which spell an absent
/// name differently: noodles reports it as no name at all, htslib hands back
/// the literal `*`. Without normalizing at decode, CRAM's unnamed records all
/// share one key and pair with each other, so the same file imports to
/// different depths depending on its container.
#[test]
fn nameless_records_never_pair_on_either_backend() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some((bam, cram, fasta)) = bam_common::nameless_overlap_pair(dir.path()) else {
        return;
    };

    // Four independent 20M reads: two at [10,30), two at [25,45).
    let mut expected = vec![0i32; 60];
    for v in &mut expected[10..25] {
        *v = 2;
    }
    for v in &mut expected[25..30] {
        *v = 4;
    }
    for v in &mut expected[30..45] {
        *v = 2;
    }

    for (label, path, reference) in [("bam", &bam, None), ("cram", &cram, Some(fasta.as_path()))] {
        let (mut backend, _genome) = Backend::open(path, reference).unwrap();
        let mut acc = Accumulators::new(1, 60);
        let mut mates = MateBuffer::default();
        let filter = DepthFilter::default();
        backend
            .fetch("nref", 0, 60, &mut |read| {
                assert!(
                    read.qname.is_empty(),
                    "{label}: qname should normalize to empty"
                );
                walk_record(
                    read,
                    0,
                    60,
                    &filter,
                    ImportMode::Depth,
                    &mut acc,
                    &mut mates,
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(acc.fields[0], expected, "{label}");
    }
}

/// `SEQ = *` (no bases, and therefore no qualities) alongside a real CIGAR:
/// the record still covers its reference span, so depth counts it, and its
/// unknown bases tally as `n`. Synthetic because samtools rejects a SAM line
/// whose `SEQ` is `*` while `QUAL` is not, and the record has to reach the
/// walk with both buffers empty.
#[test]
fn seq_star_record_counts_depth_and_tallies_n() {
    let mut read = synthetic_read("unmapped_seq", 0, 10, -1);
    read.paired = false;
    read.mate_same_contig = false;
    read.seq.clear();
    read.qual.clear();

    let filter = DepthFilter {
        min_bq: 30,
        ..DepthFilter::default()
    };
    let mut acc = Accumulators::new(FIELDS.len(), 60);
    let mut mates = MateBuffer::default();
    walk_record(
        &read,
        0,
        60,
        &filter,
        ImportMode::Composition,
        &mut acc,
        &mut mates,
    );

    assert!(acc.fields[0][10..30].iter().all(|&v| v == 1), "depth");
    assert!(acc.fields[5][10..30].iter().all(|&v| v == 1), "n");
    let bases: i32 = (1..=4).map(|f| acc.fields[f][10]).sum();
    assert_eq!(bases, 0, "no a/c/g/t tally without a sequence");
}
