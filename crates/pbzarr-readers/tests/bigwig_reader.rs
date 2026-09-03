//! POC smoke tests for `pbzarr_readers::BigWigReader`.
//!
//! Fixtures are written in-process via `bigtools::BigWigWrite` (see
//! `tests/common`), so these run without any external tooling.

mod common;

use ndarray::Array1;
use pbzarr::Genome;
use pbzarr::io::{OutputSinkMut, ValueReader};
use pbzarr_readers::BigWigReader;
use tempfile::TempDir;

#[test]
fn open_reports_contigs_and_schema() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 1000), ("chr2", 500)],
        &[("chr1", 0, 1000, 5.0), ("chr2", 0, 500, 5.0)],
        false,
    );

    let reader = BigWigReader::open(&bw).unwrap();
    let genome: &Genome = reader.contigs();

    assert_eq!(genome.len(), 2);
    let c1 = genome.get(genome.id("chr1").unwrap()).unwrap();
    let c2 = genome.get(genome.id("chr2").unwrap()).unwrap();
    assert_eq!(c1.length, 1000);
    assert_eq!(c2.length, 500);
    assert_eq!(reader.output_schema().len(), 1);

    // bigtools does not guarantee chrom order, so compare as a sorted set.
    let mut contigs = pbzarr_readers::bigwig::contigs(&bw).unwrap();
    contigs.sort();
    assert_eq!(
        contigs,
        vec![("chr1".to_owned(), 1000u64), ("chr2".to_owned(), 500u64)]
    );

    assert!(BigWigReader::open("/no/such/file.bw").is_err());
}

#[test]
fn read_into_fills_values_and_nan_gaps() {
    let dir = TempDir::new().unwrap();
    // Cover [0, 20) with 7.5; [20, 1000) is left uncovered (a gap).
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 1000)],
        &[("chr1", 0, 20, 7.5)],
        false,
    );

    let mut reader = BigWigReader::open(&bw).unwrap();
    let (start, end) = (10u64, 30u64);
    let mut buf: Array1<f32> = Array1::zeros((end - start) as usize);
    let mut outputs = [OutputSinkMut::F32(buf.view_mut())];
    reader.read_into("chr1", start, end, &mut outputs).unwrap();

    assert_eq!(buf.len(), 20);
    // Positions [10, 20) carry 7.5; [20, 30) are gaps -> NaN.
    for i in 0..10 {
        assert_eq!(buf[i], 7.5, "pos {}", start as usize + i);
    }
    for i in 10..20 {
        assert!(buf[i].is_nan(), "pos {} should be NaN", start as usize + i);
    }

    // 0-row view: contract is just "don't crash".
    let mut empty: Array1<f32> = Array1::zeros(0);
    let mut outputs = [OutputSinkMut::F32(empty.view_mut())];
    reader.read_into("chr1", 50, 50, &mut outputs).unwrap();
}

#[test]
fn unknown_contig_errors() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 100)],
        &[("chr1", 0, 100, 3.0)],
        false,
    );
    let mut reader = BigWigReader::open(&bw).unwrap();
    let mut buf: Array1<f32> = Array1::zeros(10);
    let mut outputs = [OutputSinkMut::F32(buf.view_mut())];
    assert!(reader.read_into("chrX", 0, 10, &mut outputs).is_err());
}
