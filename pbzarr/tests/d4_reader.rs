//! POC smoke tests for `pbzarr::io::d4::D4Reader`.
//!
//! Synthesizes a tiny d4 file with the system `d4tools` binary (skipped if
//! unavailable then exercises `D4Reader::open` and the `ValueReader` interface.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use ndarray::Array2;
use pbzarr::Genome;
use pbzarr::io::{D4Reader, ValueReader};
use tempfile::TempDir;

fn d4tools_available() -> bool {
    process::Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build a tiny d4 file by piping a bedGraph through `d4tools create`.
/// All intervals get the same value, so we know what every position should be.
fn synth_d4(tmpdir: &Path, name: &str, contigs: &[(&str, u64)], value: u32) -> PathBuf {
    let sizes_path = tmpdir.join(format!("{name}.sizes"));
    let mut sf = std::fs::File::create(&sizes_path).unwrap();
    for (c, l) in contigs {
        writeln!(sf, "{c}\t{l}").unwrap();
    }
    drop(sf);

    let bg_path = tmpdir.join(format!("{name}.bedgraph"));
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for (c, l) in contigs {
        writeln!(bf, "{c}\t0\t{l}\t{value}").unwrap();
    }
    drop(bf);

    let d4_path = tmpdir.join(format!("{name}.d4"));
    let status = process::Command::new("d4tools")
        .arg("create")
        .arg("--genome")
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");
    d4_path
}

#[test]
fn open_missing_file_errors() {
    assert!(D4Reader::open("/no/such/file.d4").is_err());
}

#[test]
fn open_reports_contigs_and_n_fields() {
    if !d4tools_available() {
        eprintln!("skipping open_reports_contigs_and_n_fields: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 1000), ("chr2", 500)], 5);

    let reader = D4Reader::open(&d4).unwrap();
    let genome: &Genome = reader.contigs();

    assert_eq!(genome.len(), 2);
    let c1 = genome.get(genome.id("chr1").unwrap()).unwrap();
    let c2 = genome.get(genome.id("chr2").unwrap()).unwrap();
    assert_eq!(c1.length, 1000);
    assert_eq!(c2.length, 500);
    assert_eq!(reader.n_fields(), 1);
}

#[test]
fn read_into_fills_constant_buffer() {
    if !d4tools_available() {
        eprintln!("skipping read_into_fills_constant_buffer: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 1000)], 7);

    let reader = D4Reader::open(&d4).unwrap();
    let start = 10u64;
    let end = 30u64;

    let mut buf: Array2<u32> = Array2::zeros(((end - start) as usize, reader.n_fields()));
    reader
        .read_into("chr1", start, end, buf.view_mut())
        .unwrap();

    assert_eq!(buf.shape(), &[20, 1]);
    assert!(
        buf.iter().all(|&v| v == 7),
        "expected all 7s, got {:?}",
        buf
    );
}

#[test]
fn empty_region_is_noop() {
    if !d4tools_available() {
        eprintln!("skipping empty_region_is_noop: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 100)], 3);

    let reader = D4Reader::open(&d4).unwrap();

    // 0-row view: contract is just "don't crash".
    let mut buf: Array2<u32> = Array2::zeros((0, 1));
    reader.read_into("chr1", 50, 50, buf.view_mut()).unwrap();
}

#[test]
fn fork_produces_independent_reader() {
    if !d4tools_available() {
        eprintln!("skipping fork_produces_independent_reader: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 200)], 42);

    let a = D4Reader::open(&d4).unwrap();
    let b = a.fork().unwrap();

    let mut buf_a: Array2<u32> = Array2::zeros((8, 1));
    let mut buf_b: Array2<u32> = Array2::zeros((8, 1));
    a.read_into("chr1", 0, 8, buf_a.view_mut()).unwrap();
    b.read_into("chr1", 0, 8, buf_b.view_mut()).unwrap();

    assert_eq!(buf_a, buf_b);
    assert!(buf_a.iter().all(|&v| v == 42));
}
