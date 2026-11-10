//! End-to-end: the public format drivers writing a format-0.5 root Dataset.
//! One range per axis; the single-range cases are rank-one reads.
#![allow(clippy::single_range_in_vec_init)]

mod common;

use pbzarr::dataset::{Dataset, Reference, ValidationStatus};
use pbzarr::import::{Config, Source, Target};
use pbzarr::io::ValueReader as _;
use pbzarr::{Contig, Genome};
use pbzarr_readers::{BedImportOptions, BigWigReader, from_bed_matrix, from_bigwig};
use tempfile::TempDir;

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
fn headered_bed_appends_into_its_track() {
    let dir = TempDir::new().unwrap();
    let header = ["chrom", "start", "end", "score"];
    let a = common::write_bed_bgzip_tabix(dir.path(), "a", &header, &[("chr1", 0, 4, vec!["3"])]);
    let b = common::write_bed_bgzip_tabix(dir.path(), "b", &header, &[("chr1", 2, 6, vec!["5"])]);
    let contigs = vec![Contig {
        name: "chr1".to_owned(),
        length: 8,
    }];
    let config = || Config {
        column_dim: Some("sample".to_owned()),
        ..Config::default()
    };
    let path = dir.path().join("score.pbz");
    let mut dataset = Dataset::create(&path, Reference::new(contigs.clone()).unwrap()).unwrap();
    from_bed_matrix(
        &mut dataset,
        &[Source::labeled(a, "s0")],
        Genome::new(contigs.clone()).unwrap(),
        &BedImportOptions::default(),
        config(),
    )
    .unwrap();

    let mut dataset = Dataset::open(&path).unwrap();
    from_bed_matrix(
        Target::Append(&mut dataset),
        &[Source::labeled(b, "s1")],
        Genome::new(contigs).unwrap(),
        &BedImportOptions {
            track: Some("score".to_owned()),
            dtype: Some(pbzarr::io::Dtype::U8),
            append: true,
            ..BedImportOptions::default()
        },
        config(),
    )
    .unwrap();

    let dataset = Dataset::open(&path).unwrap();
    assert_eq!(
        dataset
            .read_raw::<String>("sample", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &["s0", "s1"]
    );
    let score = dataset
        .read_raw::<u8>("score", &[0..8, 0..2])
        .unwrap()
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();
    assert_eq!(score.column(0).to_vec(), [3, 3, 3, 3, 0, 0, 0, 0]);
    assert_eq!(score.column(1).to_vec(), [0, 0, 5, 5, 5, 5, 0, 0]);
}
