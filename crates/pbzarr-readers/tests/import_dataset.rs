//! End-to-end: the public format drivers writing a format-0.5 root Dataset.
//! One range per axis; the single-range cases are rank-one reads.
#![allow(clippy::single_range_in_vec_init)]

mod common;

use std::path::{Path, PathBuf};

use pbzarr::dataset::{Dataset, Reference, ValidationStatus};
use pbzarr::import::{Config, Source};
use pbzarr::io::ValueReader as _;
use pbzarr::{Contig, Genome, PbzStore, Region};
use pbzarr_readers::{
    BedImportOptions, BigWigReader, D4Reader, from_bed_matrix, from_bigwig, from_d4,
};
use tempfile::TempDir;

fn d4_fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name)
}

fn d4_reference(path: &Path) -> (Reference, u64) {
    let genome = D4Reader::open(path).unwrap().contigs().clone();
    let total = genome.contigs().iter().map(|c| c.length).sum();
    (Reference::new(genome.contigs().to_vec()).unwrap(), total)
}

#[test]
fn bigwig_cohort_imports_into_a_dataset() {
    let dir = TempDir::new().unwrap();
    let first = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 16)],
        &[("chr1", 0, 8, 1.5)],
        false,
    );
    let second = common::write_bigwig(
        dir.path(),
        "b",
        &[("chr1", 16)],
        &[("chr1", 8, 16, 2.5)],
        false,
    );
    let reference = Reference::new(
        BigWigReader::open(&first)
            .unwrap()
            .contigs()
            .contigs()
            .to_vec(),
    )
    .unwrap();

    let path = dir.path().join("signal.pbz");
    let mut dataset = Dataset::create(&path, reference).unwrap();
    from_bigwig(
        &mut dataset,
        "signal",
        &[
            Source::labeled(first, "A".to_owned()),
            Source::labeled(second, "B".to_owned()),
        ],
        Config {
            chunk_size: Some(8),
            column_dim: Some("sample".to_owned()),
            ..Config::default()
        },
    )
    .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(dataset.validate().unwrap().status, ValidationStatus::Valid);
    assert_eq!(dataset.variables()["signal"].dims, ["position", "sample"]);
    assert_eq!(
        dataset
            .read_raw::<String>("sample", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &["A", "B"]
    );
    let values = dataset.read_raw::<f32>("signal", &[0..16, 0..2]).unwrap();
    assert_eq!(values.shape(), [16, 2]);
    let values = values.into_dimensionality::<ndarray::Ix2>().unwrap();
    for position in 0..16 {
        let (covered, gap) = if position < 8 {
            (values[[position, 0]], values[[position, 1]])
        } else {
            (values[[position, 1]], values[[position, 0]])
        };
        assert_eq!(covered, if position < 8 { 1.5 } else { 2.5 });
        assert!(gap.is_nan(), "position {position} should be missing");
    }
}

#[test]
fn d4_import_into_a_dataset_matches_the_store() {
    let dir = TempDir::new().unwrap();
    let source = d4_fixture("cohort_s0.d4");
    let (reference, total) = d4_reference(&source);
    let sources = [Source::new(source)];
    let config = || Config {
        chunk_size: Some(1_000),
        ..Config::default()
    };

    let mut store = PbzStore::create(dir.path().join("store.pbz")).unwrap();
    from_d4(&mut store, "depth", &sources, config()).unwrap();
    let track = store.track("depth").unwrap();
    let expected = track
        .read_region::<i32>(&Region {
            contig: store.genome_for("depth").unwrap().id("chr1").unwrap(),
            start: 0,
            end: total,
        })
        .unwrap();

    let path = dir.path().join("dataset.pbz");
    let mut dataset = Dataset::create(&path, reference).unwrap();
    from_d4(&mut dataset, "depth", &sources, config()).unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(dataset.variables()["depth"].dims, ["position"]);
    assert_eq!(
        dataset
            .read_raw::<i32>("depth", &[0..total])
            .unwrap()
            .as_slice()
            .unwrap(),
        expected.as_slice().unwrap()
    );
}

#[test]
fn bed_wide_track_imports_into_a_dataset() {
    let dir = TempDir::new().unwrap();
    let bed = common::write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "mapq"],
        &[
            ("chr1", 0, 4, vec!["4", "60"]),
            ("chr1", 4, 8, vec!["9", "300"]),
        ],
    );
    let contigs = vec![Contig {
        name: "chr1".to_owned(),
        length: 8,
    }];
    let path = dir.path().join("stats.pbz");
    let mut dataset = Dataset::create(&path, Reference::new(contigs.clone()).unwrap()).unwrap();
    from_bed_matrix(
        &mut dataset,
        &[Source::new(bed)],
        Genome::new(contigs).unwrap(),
        &BedImportOptions {
            track: Some("stats".to_owned()),
            ..BedImportOptions::default()
        },
        Config {
            column_dim: Some("field".to_owned()),
            ..Config::default()
        },
    )
    .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(dataset.variables()["stats"].dims, ["position", "field"]);
    assert_eq!(
        dataset
            .read_raw::<String>("field", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &["cov", "mapq"]
    );
    let stats = dataset
        .read_raw::<u16>("stats", &[0..8, 0..2])
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    for (position, row) in stats.rows().into_iter().enumerate() {
        let expected = if position < 4 {
            vec![4, 60]
        } else {
            vec![9, 300]
        };
        assert_eq!(row.to_vec(), expected);
    }
}

#[test]
fn dataset_import_rejects_scales() {
    let dir = TempDir::new().unwrap();
    let source = d4_fixture("cohort_s0.d4");
    let (reference, _) = d4_reference(&source);
    let path = dir.path().join("dataset.pbz");
    let mut dataset = Dataset::create(&path, reference).unwrap();

    let error = from_d4(
        &mut dataset,
        "depth",
        &[Source::new(source)],
        Config {
            scales: vec![16],
            ..Config::default()
        },
    )
    .err()
    .expect("a format-0.5 import must reject scales")
    .to_string();

    assert!(error.contains("scales are not supported"), "{error}");
    assert!(
        !Dataset::open(&path)
            .unwrap()
            .variables()
            .contains_key("depth")
    );
}
