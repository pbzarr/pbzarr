//! POC smoke tests for `pbzarr_readers::D4Reader`.
//!
//! Uses the committed fixtures under `fixtures/d4/`, so these run without
//! any external tooling.

use std::path::{Path, PathBuf};

use ndarray::Array1;
use pbzarr::Genome;
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
    let genome: &Genome = reader.contigs();

    assert_eq!(genome.len(), 2);
    let c1 = genome.get(genome.id("chr1").unwrap()).unwrap();
    let c2 = genome.get(genome.id("chr2").unwrap()).unwrap();
    assert_eq!(c1.length, 1000);
    assert_eq!(c2.length, 500);
    assert_eq!(reader.output_schema().len(), 1);

    let contigs = pbzarr_readers::d4::contigs(&d4).unwrap();
    assert_eq!(
        contigs,
        vec![("chr1".to_owned(), 1000u64), ("chr2".to_owned(), 500u64)]
    );

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
