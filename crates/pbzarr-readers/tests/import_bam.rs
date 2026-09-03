//! End-to-end: import the `bam_common` fixture (BAM and its CRAM twin) via
//! `from_bam`, read back through the store, and check against the
//! hand-computed depth vector in `bam_common::expected_default` (see
//! `bam_common::fixture` for its per-record derivation).

mod bam_common;

use std::path::Path;

use pbzarr::import::{Config, Source};
use pbzarr::io::Dtype;
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{DepthFilter, ImportMode, from_bam};
use tempfile::TempDir;

/// Count chunk data files under an array directory (everything but
/// `zarr.json`). zarrs elides chunks equal to the fill value, so an all-gap
/// span leaves nothing on disk (mirrors `import_bigwig.rs`'s helper).
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

fn region(store: &PbzStore, track: &str, contig: &str, start: u64, end: u64) -> Region {
    Region {
        contig: store.genome_for(track).unwrap().id(contig).unwrap(),
        start,
        end,
    }
}

#[test]
fn depth_single_source_scalar_roundtrip() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let store_path = dir.path().join("depth.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_bam(
        &mut store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config {
            chunk_size: Some(100),
            ..Config::default()
        },
    )
    .unwrap();

    let track = store.track("depth").unwrap();
    assert_eq!(track.rank(), 1);
    assert_eq!(track.dtype(), Dtype::I32);

    let ref1 = track
        .read_region::<i32>(&region(&store, "depth", "ref1", 0, 100))
        .unwrap();
    assert_eq!(
        ref1.into_raw_vec_and_offset().0,
        bam_common::expected_default().to_vec()
    );

    let ref2 = track
        .read_region::<i32>(&region(&store, "depth", "ref2", 0, 60))
        .unwrap();
    assert!(ref2.iter().all(|&v| v == 0));

    // ref1 occupies flat chunk 0 ([0,100)), ref2 occupies flat chunk 1
    // ([100,160)); with chunk_size=100 they land in separate physical
    // chunks, so ref2's all-zero chunk should never be written.
    let values_dir = store_path.join("depth").join("values");
    assert_eq!(count_chunk_files(&values_dir), 1);

    // `description`/`source` land on the track's `perbase:` block (F8); read
    // it straight off disk rather than pinning the exact string content.
    let zarr_json = std::fs::read_to_string(store_path.join("depth").join("zarr.json")).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&zarr_json).unwrap();
    let attrs = doc.get("attributes").unwrap();
    assert!(attrs.get("perbase:description").is_some());
    assert!(attrs.get("perbase:source").is_some());
}

#[test]
fn depth_two_sources_cohort() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let store_path = dir.path().join("cohort.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_bam(
        &mut store,
        "depth",
        &[
            Source::labeled(&fx.bam, "s1"),
            Source::labeled(&fx.bam, "s2"),
        ],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config::default(),
    )
    .unwrap();

    let track = store.track("depth").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.columns_count().unwrap(), 2);
    assert_eq!(track.column_dim(), Some("sample"));

    let data = track
        .read_region::<i32>(&region(&store, "depth", "ref1", 0, 100))
        .unwrap();
    let expected = bam_common::expected_default();
    for pos in 0..100 {
        assert_eq!(data[[pos, 0]], expected[pos]);
        assert_eq!(data[[pos, 1]], expected[pos]);
    }

    // D-1: the 2D gate is `n_sources > 1 || column_dim.is_some()`, so an
    // explicit `column_dim` on a single source still produces a 2D track
    // (with one column), matching d4/bigwig/bed.
    let single_path = dir.path().join("single.pbz");
    let mut single_store = PbzStore::create(&single_path).unwrap();
    from_bam(
        &mut single_store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .unwrap();

    let single = single_store.track("depth").unwrap();
    assert_eq!(single.rank(), 2);
    assert_eq!(single.columns_count().unwrap(), 1);
    assert_eq!(single.column_dim(), Some("sample"));
}

#[test]
fn cram_equals_bam() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let bam_path = dir.path().join("bam.pbz");
    let mut bam_store = PbzStore::create(&bam_path).unwrap();
    from_bam(
        &mut bam_store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config::default(),
    )
    .unwrap();

    let cram_path = dir.path().join("cram.pbz");
    let mut cram_store = PbzStore::create(&cram_path).unwrap();
    from_bam(
        &mut cram_store,
        "depth",
        &[Source::new(&fx.cram)],
        ImportMode::Depth,
        DepthFilter::default(),
        Some(fx.fasta.clone()),
        Config::default(),
    )
    .unwrap();

    let bam_data = bam_store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&bam_store, "depth", "ref1", 0, 100))
        .unwrap();
    let cram_data = cram_store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&cram_store, "depth", "ref1", 0, 100))
        .unwrap();
    assert_eq!(bam_data, cram_data);
}

#[test]
fn composition_counts() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let store_path = dir.path().join("comp.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    let filter = DepthFilter {
        min_bq: 20,
        ..DepthFilter::default()
    };
    from_bam(
        &mut store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Composition,
        filter,
        None,
        Config::default(),
    )
    .unwrap();

    // r6 (5M1I5M at pos 50, all-C) leaves a single insertion anchored at ref
    // pos 54, the last position its M block consumed before the gap.
    let ins = store
        .track("depth_ins")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_ins", "ref1", 0, 100))
        .unwrap();
    assert_eq!(ins[54], 1);
    assert!(ins.iter().enumerate().all(|(i, &v)| i == 54 || v == 0));

    // r2 (10M5D10M at pos 40, all-A) leaves a deletion run over [50,55).
    let del = store
        .track("depth_del")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_del", "ref1", 0, 100))
        .unwrap();
    for pos in 50..55 {
        assert_eq!(del[pos], 1);
    }

    // r7 (10M at pos 12, all-G) has a BQ5 base at read offset 3 (ref pos 15);
    // with min_bq=20 that base is gated to the N field instead of G.
    let n = store
        .track("depth_n")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_n", "ref1", 0, 100))
        .unwrap();
    assert_eq!(n[15], 1);

    let g = store
        .track("depth_g")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_g", "ref1", 0, 100))
        .unwrap();
    assert_eq!(g[12], 1); // r7's first base, BQ30, clears the gate
    assert_eq!(g[15], 0); // gated to N instead

    // r1b's exclusive territory [30,40) (past r1a/r5's [10,30) and before
    // r2's M[40,50)): only one all-A read covers it.
    let a = store
        .track("depth_a")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_a", "ref1", 0, 100))
        .unwrap();
    assert_eq!(a[35], 1);

    // r6's Match bases (all-C) start at ref pos 50.
    let c = store
        .track("depth_c")
        .unwrap()
        .read_region::<i32>(&region(&store, "depth_c", "ref1", 0, 100))
        .unwrap();
    assert_eq!(c[50], 1);
}

#[test]
fn chunk_boundary_read() {
    let fx = bam_common::fixture();
    let dir = TempDir::new().unwrap();

    let default_path = dir.path().join("default.pbz");
    let mut default_store = PbzStore::create(&default_path).unwrap();
    from_bam(
        &mut default_store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config::default(),
    )
    .unwrap();

    let chunked_path = dir.path().join("chunked.pbz");
    let mut chunked_store = PbzStore::create(&chunked_path).unwrap();
    from_bam(
        &mut chunked_store,
        "depth",
        &[Source::new(&fx.bam)],
        ImportMode::Depth,
        DepthFilter::default(),
        None,
        Config {
            chunk_size: Some(16),
            ..Config::default()
        },
    )
    .unwrap();

    let default_data = default_store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&default_store, "depth", "ref1", 0, 100))
        .unwrap();
    let chunked_data = chunked_store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&chunked_store, "depth", "ref1", 0, 100))
        .unwrap();
    assert_eq!(default_data, chunked_data);
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

    let import = |name: &str, chunk_size: usize| {
        let path = dir.path().join(name);
        let mut store = PbzStore::create(&path).unwrap();
        from_bam(
            &mut store,
            "comp",
            &[Source::new(&bam)],
            ImportMode::Composition,
            DepthFilter::default(),
            None,
            Config {
                chunk_size: Some(chunk_size),
                ..Config::default()
            },
        )
        .unwrap();
        let ins = store
            .track("comp_ins")
            .unwrap()
            .read_region::<i32>(&region(&store, "comp_ins", "iref", 0, 40))
            .unwrap();
        ins.into_raw_vec_and_offset().0
    };

    let small = import("ins_small.pbz", 5);
    let big = import("ins_big.pbz", 20);
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
    let mut store = PbzStore::create(dir.path().join("missing_index.pbz")).unwrap();
    assert!(
        from_bam(
            &mut store,
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
    let mut store = PbzStore::create(dir.path().join("no_reference.pbz")).unwrap();
    assert!(
        from_bam(
            &mut store,
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
