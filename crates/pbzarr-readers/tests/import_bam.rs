//! End-to-end: import the `bam_common` fixture (BAM and its CRAM twin) via
//! `from_bam`, read back through the Dataset, and check against the
//! hand-computed depth vector in `bam_common::expected_default` (see
//! `bam_common::fixture` for its per-record derivation).

mod bam_common;

use std::path::Path;

use ndarray::ArrayD;
use pbzarr::Dataset;
use pbzarr::import::{Config, Source};
use pbzarr_readers::{DepthFilter, ImportMode, from_bam};
use tempfile::TempDir;

/// Count chunk data files under an array directory (everything but
/// `zarr.json`). zarrs elides chunks equal to the fill value, so an all-gap
/// span leaves nothing on disk.
fn count_chunk_files(array_dir: &Path) -> usize {
    fn walk(dir: &Path, n: &mut usize) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, n);
            } else if path.file_name().and_then(|s| s.to_str()) != Some("zarr.json") {
                *n += 1;
            }
        }
    }
    let mut n = 0;
    walk(array_dir, &mut n);
    n
}

fn read(store: &Path, track: &str, region: &str) -> ArrayD<i32> {
    Dataset::open(store)
        .unwrap()
        .read_region::<i32>(track, &region.parse().unwrap())
        .unwrap()
}

fn import(store: &Path, track: &str, sources: &[Source], mode: ImportMode, config: Config) {
    from_bam(
        store,
        track,
        sources,
        mode,
        DepthFilter::default(),
        None,
        config,
    )
    .unwrap();
}

#[test]
fn depth_single_source_scalar_roundtrip() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let store = dir.path().join("depth.pbz");
    import(
        &store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        Config {
            chunk_size: Some(100),
            ..Config::default()
        },
    );

    assert_eq!(
        Dataset::open(&store).unwrap().variables()["depth"].dims,
        ["position"]
    );
    assert_eq!(
        read(&store, "depth", "ref1").into_raw_vec_and_offset().0,
        bam_common::expected_default().to_vec()
    );
    assert!(read(&store, "depth", "ref2").iter().all(|&v| v == 0));

    // ref1 occupies flat chunk 0 ([0,100)), ref2 occupies flat chunk 1
    // ([100,160)); with chunk_size=100 they land in separate physical
    // chunks, so ref2's all-zero chunk should never be written.
    assert_eq!(count_chunk_files(&store.join("depth")), 1);
}

#[test]
fn depth_two_sources_cohort() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let store = dir.path().join("cohort.pbz");
    import(
        &store,
        "depth",
        &[
            Source::labeled(&fx.bam, "s1"),
            Source::labeled(&fx.bam, "s2"),
        ],
        ImportMode::Depth,
        Config::default(),
    );

    let ds = Dataset::open(&store).unwrap();
    assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
    let data = read(&store, "depth", "ref1");
    assert_eq!(data.shape(), [100, 2]);
    let expected = bam_common::expected_default();
    for pos in 0..100 {
        assert_eq!(data[[pos, 0]], expected[pos]);
        assert_eq!(data[[pos, 1]], expected[pos]);
    }

    // An explicit `column_dim` on a single source still produces a 2D track
    // with one column.
    let single = dir.path().join("single.pbz");
    import(
        &single,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    );
    let ds = Dataset::open(&single).unwrap();
    assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
    assert_eq!(ds.variables()["depth"].shape, [160, 1]);
}

#[test]
fn cram_equals_bam() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let bam = dir.path().join("bam.pbz");
    import(
        &bam,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        Config::default(),
    );

    let cram = dir.path().join("cram.pbz");
    from_bam(
        cram.as_path(),
        "depth",
        &[Source::new(&fx.cram)],
        ImportMode::Depth,
        DepthFilter::default(),
        Some(fx.fasta.clone()),
        Config::default(),
    )
    .unwrap();

    assert_eq!(read(&bam, "depth", "ref1"), read(&cram, "depth", "ref1"));
}

#[test]
fn composition_counts() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let store = dir.path().join("comp.pbz");
    from_bam(
        store.as_path(),
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Composition,
        DepthFilter {
            min_bq: 20,
            ..DepthFilter::default()
        },
        None,
        Config::default(),
    )
    .unwrap();

    // r6 (5M1I5M at pos 50, all-C) leaves a single insertion anchored at ref
    // pos 54, the last position its M block consumed before the gap.
    let ins = read(&store, "depth_ins", "ref1");
    assert_eq!(ins[54], 1);
    assert!(ins.iter().enumerate().all(|(i, &v)| i == 54 || v == 0));

    // r2 (10M5D10M at pos 40, all-A) leaves a deletion run over [50,55).
    let del = read(&store, "depth_del", "ref1");
    for pos in 50..55 {
        assert_eq!(del[pos], 1);
    }

    // r7 (10M at pos 12, all-G) has a BQ5 base at read offset 3 (ref pos 15);
    // with min_bq=20 that base is gated to the N field instead of G.
    assert_eq!(read(&store, "depth_n", "ref1")[15], 1);
    let g = read(&store, "depth_g", "ref1");
    assert_eq!(g[12], 1);
    assert_eq!(g[15], 0);

    // r1b's exclusive territory [30,40) (past r1a/r5's [10,30) and before
    // r2's M[40,50)): only one all-A read covers it.
    assert_eq!(read(&store, "depth_a", "ref1")[35], 1);

    // r6's Match bases (all-C) start at ref pos 50.
    assert_eq!(read(&store, "depth_c", "ref1")[50], 1);
}

#[test]
fn chunk_boundary_read() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let default = dir.path().join("default.pbz");
    import(
        &default,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        Config::default(),
    );
    let chunked = dir.path().join("chunked.pbz");
    import(
        &chunked,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        Config {
            chunk_size: Some(16),
            ..Config::default()
        },
    );

    let chunked_data = read(&chunked, "depth", "ref1");
    assert_eq!(read(&default, "depth", "ref1"), chunked_data);
    assert_eq!(
        chunked_data.into_raw_vec_and_offset().0,
        bam_common::expected_default().to_vec()
    );
}

/// An insertion's anchor must sit inside the read's own reference span, so
/// the chunked import (whose per-chunk query only returns reads overlapping
/// that chunk) records it identically at any chunk size. A trailing `I`
/// anchored at `ref_pos` would land one past the span and vanish whenever the
/// chunk grid cuts right after the read.
#[test]
fn insertion_anchor_is_chunk_size_independent() {
    let bam = bam_common::ins_anchor_bam();
    let dir = TempDir::new().unwrap();

    let ins = |name: &str, chunk_size: usize| {
        let store = dir.path().join(name);
        import(
            &store,
            "comp",
            &[Source::new(&bam)],
            ImportMode::Composition,
            Config {
                chunk_size: Some(chunk_size),
                ..Config::default()
            },
        );
        read(&store, "comp_ins", "iref").into_raw_vec_and_offset().0
    };

    let small = ins("ins_small.pbz", 5);
    let big = ins("ins_big.pbz", 20);
    assert_eq!(small, big);

    // r10 is `5M1I` at pos 10: last ref-consuming position is 14.
    assert_eq!(small[14], 1);
    // r11 is `1I5M` at pos 20: no ref op precedes the insertion, so it stays
    // anchored at the read's start.
    assert_eq!(small[20], 1);
    assert_eq!(small.iter().sum::<i32>(), 2);
}

#[test]
fn errors() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    // Missing index: copy the BAM alone, no .bai alongside it.
    let unindexed = dir.path().join("unindexed.bam");
    std::fs::copy(&fx.bam, &unindexed).unwrap();
    let missing_index = dir.path().join("missing_index.pbz");
    assert!(
        from_bam(
            missing_index.as_path(),
            "depth",
            &[Source::new(&unindexed)],
            ImportMode::Depth,
            DepthFilter::default(),
            None,
            Config::default(),
        )
        .is_err()
    );

    // CRAM without an explicit reference.
    let no_reference = dir.path().join("no_reference.pbz");
    assert!(
        from_bam(
            no_reference.as_path(),
            "depth",
            &[Source::new(&fx.cram)],
            ImportMode::Depth,
            DepthFilter::default(),
            None,
            Config::default(),
        )
        .is_err()
    );
}
