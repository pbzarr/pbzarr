//! End-to-end: synthesize one multi-column bgzipped+tabix BED, import all value
//! columns to their own tracks via `from_bed_multi`, read each back. Skipped if
//! bgzip/tabix are unavailable.

mod common;

use ndarray::{Ix1, Ix2};
use pbzarr::genome::{Contig, Genome};
use pbzarr::import::Config;
use pbzarr::io::Dtype;
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, BedSource, from_bed_matrix, from_bed_multi,
};
use tempfile::TempDir;

use common::{htslib_available, write_bed_bgzip_tabix};

fn genome(cs: &[(&str, u64)]) -> Genome {
    Genome::new(
        cs.iter()
            .map(|(n, l)| Contig {
                name: (*n).to_owned(),
                length: *l,
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn multi_column_mixed_dtype_roundtrip() {
    if !htslib_available() {
        eprintln!(
            "skip import_bed_multi::multi_column_mixed_dtype_roundtrip: bgzip/tabix not on PATH"
        );
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "score", "mask"],
        &[
            ("chr1", 0, 20, vec!["4", "1.5", "1"]),
            ("chr1", 20, 40, vec!["9", "2.5", "0"]),
        ],
    );
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();

    // Two name selectors (default track names) + one index selector.
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("score", Dtype::F32),
        BedColumnSpec::indexed(5, Dtype::Bool, "mask"),
    ]);
    from_bed_multi(
        &mut store,
        &bed,
        &schema,
        genome(&[("chr1", 40)]),
        Config::default(),
    )
    .unwrap();

    // Every track owns an identical genome; contig ids share the same index.
    let g = store.genome_for("cov").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 40,
    };

    let cov = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(cov.iter().take(20).all(|&v| v == 4));
    assert!(cov.iter().skip(20).all(|&v| v == 9));

    let score = store
        .track("score")
        .unwrap()
        .read_region::<f32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(score.iter().take(20).all(|&v| v == 1.5));
    assert!(score.iter().skip(20).all(|&v| v == 2.5));

    let mask = store
        .track("mask")
        .unwrap()
        .read_region::<bool>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(mask.iter().take(20).all(|&v| v));
    assert!(mask.iter().skip(20).all(|&v| !v));
}

#[test]
fn multi_source_matrix_import_reconciles_reordered_headers() {
    if !htslib_available() {
        eprintln!("skip matrix import test: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let a = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage", "score"],
        &[("chr1", 0, 8, vec!["4", "1.5"])],
    );
    let b = write_bed_bgzip_tabix(
        dir.path(),
        "b",
        &["chrom", "start", "end", "score", "coverage"],
        &[("chr1", 0, 8, vec!["2.5", "9"])],
    );
    let mut store = PbzStore::create(dir.path().join("matrix.pbz")).unwrap();
    from_bed_matrix(
        &mut store,
        &[
            BedSource {
                path: a,
                column_label: Some("A".into()),
            },
            BedSource {
                path: b,
                column_label: Some("B".into()),
            },
        ],
        genome(&[("chr1", 8)]),
        &BedImportOptions::default(),
        Config {
            chunk_size: Some(4),
            column_chunk_size: Some(1),
            shard_size: Some(4),
            shard_column_size: Some(2),
            workers: 2,
            ..Config::default()
        },
    )
    .unwrap();
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let coverage = store
        .track("coverage")
        .unwrap()
        .read_region::<u8>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let score = store
        .track("score")
        .unwrap()
        .read_region::<f32>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert!(
        coverage
            .rows()
            .into_iter()
            .all(|row| row.to_vec() == vec![4, 9])
    );
    assert!(
        score
            .rows()
            .into_iter()
            .all(|row| row.to_vec() == vec![1.5, 2.5])
    );
}

#[test]
fn multi_column_chunk_and_contig_boundaries() {
    if !htslib_available() {
        eprintln!(
            "skip import_bed_multi::multi_column_chunk_and_contig_boundaries: bgzip/tabix not on PATH"
        );
        return;
    }
    let dir = TempDir::new().unwrap();
    // chr1 run spans the chunk boundary at 100; chr2 exercises the contig straddle.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "flag"],
        &[
            ("chr1", 0, 250, vec!["5", "1"]),
            ("chr2", 0, 100, vec!["8", "0"]),
        ],
    );
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);
    let config = Config {
        chunk_size: Some(100),
        ..Config::default()
    };
    from_bed_multi(
        &mut store,
        &bed,
        &schema,
        genome(&[("chr1", 250), ("chr2", 100)]),
        config,
    )
    .unwrap();

    let g = store.genome_for("cov").unwrap();
    let r1 = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 250,
    };
    let r2 = Region {
        contig: g.id("chr2").unwrap(),
        start: 0,
        end: 100,
    };

    let cov1 = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&r1)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(
        cov1.iter().all(|&v| v == 5),
        "run uniform across chunk boundary"
    );
    let cov2 = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&r2)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(cov2.iter().all(|&v| v == 8));

    let flag1 = store
        .track("flag")
        .unwrap()
        .read_region::<bool>(&r1)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(flag1.iter().all(|&v| v));
    let flag2 = store
        .track("flag")
        .unwrap()
        .read_region::<bool>(&r2)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(flag2.iter().all(|&v| !v));
}

#[test]
fn uncovered_positions_read_back_as_zero() {
    if !htslib_available() {
        eprintln!(
            "skip import_bed_multi::uncovered_positions_read_back_as_zero: bgzip/tabix not on PATH"
        );
        return;
    }
    let dir = TempDir::new().unwrap();
    // chr1 is length 40 but records cover only [0,10) and [20,30); the gap
    // [10,20) and the tail [30,40) are never written and must read as zero.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "flag"],
        &[
            ("chr1", 0, 10, vec!["5", "1"]),
            ("chr1", 20, 30, vec!["7", "0"]),
        ],
    );
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);
    from_bed_multi(
        &mut store,
        &bed,
        &schema,
        genome(&[("chr1", 40)]),
        Config::default(),
    )
    .unwrap();

    let g = store.genome_for("cov").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 40,
    };

    let cov = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(cov.iter().take(10).all(|&v| v == 5));
    assert!(
        cov.iter().skip(10).take(10).all(|&v| v == 0),
        "gap reads as zero"
    );
    assert!(cov.iter().skip(20).take(10).all(|&v| v == 7));
    assert!(cov.iter().skip(30).all(|&v| v == 0), "tail reads as zero");

    let flag = store
        .track("flag")
        .unwrap()
        .read_region::<bool>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(flag.iter().take(10).all(|&v| v));
    assert!(
        flag.iter().skip(10).all(|&v| !v),
        "uncovered + true-zero both read false"
    );
}

#[test]
fn population_failure_leaves_all_tracks_unpublished() {
    if !htslib_available() {
        eprintln!(
            "skip import_bed_multi::population_failure_leaves_all_tracks_unpublished: bgzip/tabix not on PATH"
        );
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "invalid",
        &["chrom", "start", "end", "cov", "flag"],
        &[("chr1", 0, 10, vec!["not-an-int", "1"])],
    );
    let path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);

    let result = from_bed_multi(
        &mut store,
        &bed,
        &schema,
        genome(&[("chr1", 10)]),
        Config::default(),
    );

    assert!(result.is_err());
    assert!(store.track("cov").is_none());
    assert!(store.track("flag").is_none());
    let reopened = PbzStore::open(path).unwrap();
    assert!(reopened.track("cov").is_none());
    assert!(reopened.track("flag").is_none());
}
