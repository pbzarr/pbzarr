//! POC smoke tests for `pbzarr_readers::BigWigReader`.
//!
//! Fixtures are written in-process via `bigtools::BigWigWrite` (see
//! `tests/common`), so these run without any external tooling.

mod common;

use ndarray::Array2;
use pbzarr::Genome;
use pbzarr::io::ValueReader;
use pbzarr_readers::BigWigReader;
use tempfile::TempDir;

#[test]
fn open_missing_file_errors() {
    assert!(BigWigReader::open("/no/such/file.bw").is_err());
}

#[test]
fn open_reports_contigs_and_n_fields() {
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
    assert_eq!(reader.n_fields(), 1);
}

#[test]
fn read_into_fills_values_and_zero_gaps() {
    let dir = TempDir::new().unwrap();
    // Cover [0, 20) with 7.5; [20, 1000) is left uncovered (a gap).
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 1000)],
        &[("chr1", 0, 20, 7.5)],
        false,
    );

    let reader = BigWigReader::open(&bw).unwrap();
    let (start, end) = (10u64, 30u64);
    let mut buf: Array2<f32> = Array2::zeros(((end - start) as usize, reader.n_fields()));
    reader
        .read_into("chr1", start, end, buf.view_mut())
        .unwrap();

    assert_eq!(buf.shape(), &[20, 1]);
    // Positions [10, 20) carry 7.5; [20, 30) are gaps -> 0.
    for i in 0..10 {
        assert_eq!(buf[[i, 0]], 7.5, "pos {}", start as usize + i);
    }
    for i in 10..20 {
        assert_eq!(buf[[i, 0]], 0.0, "pos {} should be 0", start as usize + i);
    }
}

#[test]
fn empty_region_is_noop() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 100)],
        &[("chr1", 0, 100, 3.0)],
        false,
    );

    let reader = BigWigReader::open(&bw).unwrap();
    // 0-row view: contract is just "don't crash".
    let mut buf: Array2<f32> = Array2::zeros((0, 1));
    reader.read_into("chr1", 50, 50, buf.view_mut()).unwrap();
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
    let reader = BigWigReader::open(&bw).unwrap();
    let mut buf: Array2<f32> = Array2::zeros((10, 1));
    assert!(reader.read_into("chrX", 0, 10, buf.view_mut()).is_err());
}

#[test]
fn fork_produces_independent_reader() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 200)],
        &[("chr1", 0, 200, 42.0)],
        false,
    );

    let a = BigWigReader::open(&bw).unwrap();
    let b = a.fork().unwrap();

    let mut buf_a: Array2<f32> = Array2::zeros((8, 1));
    let mut buf_b: Array2<f32> = Array2::zeros((8, 1));
    a.read_into("chr1", 0, 8, buf_a.view_mut()).unwrap();
    b.read_into("chr1", 0, 8, buf_b.view_mut()).unwrap();

    assert_eq!(buf_a, buf_b);
    assert!(buf_a.iter().all(|&v| v == 42.0));
}
