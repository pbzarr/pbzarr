//! POC smoke tests for `pbzarr_readers::D4Reader`.
//!
//! Uses the committed fixtures under `fixtures/d4/`, so these run without
//! any external tooling.

use std::path::{Path, PathBuf};

use ndarray::Array1;
use pbzarr::Reference;
use pbzarr::io::{OutputSinkMut, ValueReader};
use pbzarr_readers::D4Reader;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name)
}

#[test]
fn open_reports_contigs_and_schema() {
    let d4 = fixture("two_contig_const7.d4");

    let reader = D4Reader::open(&d4).unwrap();
    let genome: &Reference = reader.contigs();

    assert_eq!(genome.len(), 2);
    let c1 = genome.get(genome.id("chr1").unwrap()).unwrap();
    let c2 = genome.get(genome.id("chr2").unwrap()).unwrap();
    assert_eq!(c1.length, 1000);
    assert_eq!(c2.length, 500);
    assert_eq!(reader.output_schema().len(), 1);

    assert!(D4Reader::open("/no/such/file.d4").is_err());
}

#[test]
fn read_into_fills_constant_buffer() {
    let d4 = fixture("two_contig_const7.d4");

    let mut reader = D4Reader::open(&d4).unwrap();
    let start = 10u64;
    let end = 30u64;

    let mut buf: Array1<i32> = Array1::zeros((end - start) as usize);
    let mut outputs = [OutputSinkMut::I32(buf.view_mut())];
    reader.read_into("chr1", start, end, &mut outputs).unwrap();

    assert_eq!(buf.len(), 20);
    assert!(
        buf.iter().all(|&v| v == 7),
        "expected all 7s, got {:?}",
        buf
    );

    // 0-row view: contract is just "don't crash".
    let mut empty: Array1<i32> = Array1::zeros(0);
    let mut outputs = [OutputSinkMut::I32(empty.view_mut())];
    reader.read_into("chr1", 50, 50, &mut outputs).unwrap();

    let mut buf: Array1<i32> = Array1::zeros(10);
    let mut outputs = [OutputSinkMut::I32(buf.view_mut())];
    assert!(reader.read_into("chrX", 0, 10, &mut outputs).is_err());
}

// cohort_s1.d4 holds 101..=150 on chr1, so every base needs the secondary table.
fn cohort_s1_value(pos: u64) -> i32 {
    101 + ((pos / 10) % 50) as i32
}

fn read_region(reader: &mut D4Reader, start: u64, end: u64) -> Array1<i32> {
    let mut buf: Array1<i32> = Array1::zeros((end - start) as usize);
    let mut outputs = [OutputSinkMut::I32(buf.view_mut())];
    reader.read_into("chr1", start, end, &mut outputs).unwrap();
    buf
}

#[test]
fn reused_handle_reads_backward() {
    let mut reader = D4Reader::open(fixture("cohort_s1.d4")).unwrap();
    for (start, end) in [(9_000, 10_000), (5_000, 6_000), (1_000, 2_000), (0, 10_000)] {
        let buf = read_region(&mut reader, start, end);
        let expected: Array1<i32> = (start..end).map(cohort_s1_value).collect();
        assert_eq!(buf, expected, "region {start}..{end}");
    }
}
