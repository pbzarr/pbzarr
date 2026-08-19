//! POC smoke tests for `pbzarr_readers::D4Reader`.
//!
//! Synthesizes a tiny d4 file with the system `d4tools` binary (skipped if
//! unavailable then exercises `D4Reader::open` and the `ValueReader` interface.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use ndarray::Array1;
use pbzarr::Genome;
use pbzarr::io::{OutputSinkMut, ValueReader};
use pbzarr_readers::D4Reader;
use tempfile::TempDir;

/// True if `d4tools` is on PATH. With `PBZ_REQUIRE_TOOLS` set (CI), a
/// missing tool panics instead of letting callers self-skip.
fn d4tools_available() -> bool {
    let ok = process::Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok && std::env::var_os("PBZ_REQUIRE_TOOLS").is_some() {
        panic!("PBZ_REQUIRE_TOOLS is set but d4tools is not on PATH");
    }
    ok
}

/// Build a tiny d4 file by piping a bedGraph through `d4tools create`.
/// All intervals get the same value, so we know what every position should be.
fn synth_d4(tmpdir: &Path, name: &str, contigs: &[(&str, u64)], value: i32) -> PathBuf {
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
fn open_reports_contigs_and_schema() {
    if !d4tools_available() {
        eprintln!("skipping open_reports_contigs_and_schema: d4tools not in PATH");
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
    assert_eq!(reader.output_schema().len(), 1);
}

#[test]
fn contigs_reads_header_pairs() {
    if !d4tools_available() {
        eprintln!("skipping contigs_reads_header_pairs: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 1000), ("chr2", 500)], 5);

    let contigs = pbzarr_readers::d4::contigs(&d4).unwrap();
    assert_eq!(
        contigs,
        vec![("chr1".to_owned(), 1000u64), ("chr2".to_owned(), 500u64)]
    );
}

#[test]
fn read_into_fills_constant_buffer() {
    if !d4tools_available() {
        eprintln!("skipping read_into_fills_constant_buffer: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 1000)], 7);

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
}

#[test]
fn empty_region_is_noop() {
    if !d4tools_available() {
        eprintln!("skipping empty_region_is_noop: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 100)], 3);

    let mut reader = D4Reader::open(&d4).unwrap();

    // 0-row view: contract is just "don't crash".
    let mut buf: Array1<i32> = Array1::zeros(0);
    let mut outputs = [OutputSinkMut::I32(buf.view_mut())];
    reader.read_into("chr1", 50, 50, &mut outputs).unwrap();
}

#[test]
fn fork_produces_independent_reader() {
    if !d4tools_available() {
        eprintln!("skipping fork_produces_independent_reader: d4tools not in PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = synth_d4(dir.path(), "a", &[("chr1", 200)], 42);

    let mut a = D4Reader::open(&d4).unwrap();
    let mut b = a.fork().unwrap();

    let mut buf_a: Array1<i32> = Array1::zeros(8);
    let mut buf_b: Array1<i32> = Array1::zeros(8);
    a.read_into("chr1", 0, 8, &mut [OutputSinkMut::I32(buf_a.view_mut())])
        .unwrap();
    b.read_into("chr1", 0, 8, &mut [OutputSinkMut::I32(buf_b.view_mut())])
        .unwrap();

    assert_eq!(buf_a, buf_b);
    assert!(buf_a.iter().all(|&v| v == 42));
}
