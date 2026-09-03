use std::fs::File;
use std::io::Read;

use noodles_bgzf as bgzf;
use tempfile::TempDir;

mod common;
use common::{build_store, run_pbz, stdout_of};

#[test]
fn integer_track_streams_collapsed_rows_in_genome_order() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());

    let ambiguous = run_pbz(["view".into(), store.as_os_str().to_owned()]);
    assert!(!ambiguous.status.success());
    let error = String::from_utf8_lossy(&ambiguous.stderr);
    assert!(error.contains("store has 2 tracks"), "{error}");

    let output = run_pbz(["view".into(), store.into_os_string(), "depth".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\tdepth\n\
         chr1\t0\t3\t5\n\
         chr1\t3\t5\t7\n\
         chr1\t5\t10\t0\n\
         chr2\t0\t6\t0\n"
    );
}

#[test]
fn float_track_skips_all_missing_positions() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["view".into(), store.into_os_string(), "af".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t2\t4\t0.5\t1\n\
         chr1\t4\t6\t0.25\t1\n"
    );
}

#[test]
fn column_subset_and_no_header() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz([
        "view".into(),
        "--no-header".into(),
        "-c".into(),
        "s2".into(),
        store.into_os_string(),
        "af".into(),
    ]);
    assert_eq!(stdout_of(&output), "chr1\t2\t6\t1\n");
}

#[test]
fn gz_output_writes_bgzf_with_identical_content() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let gz = dir.path().join("out.bedgraph.gz");
    let plain = stdout_of(&run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "depth".into(),
    ]));
    let output = run_pbz([
        "view".into(),
        "-o".into(),
        gz.as_os_str().to_owned(),
        store.into_os_string(),
        "depth".into(),
    ]);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());

    let raw = std::fs::read(&gz).unwrap();
    assert_eq!(&raw[..2], &[0x1f, 0x8b], "not gzip-framed");
    let mut reader = bgzf::io::Reader::new(File::open(&gz).unwrap());
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, plain);
}

#[test]
fn threads_option_does_not_change_output() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let default_run = stdout_of(&run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "depth".into(),
    ]));
    let capped = stdout_of(&run_pbz([
        "view".into(),
        "-t".into(),
        "1".into(),
        store.into_os_string(),
        "depth".into(),
    ]));
    assert_eq!(capped, default_run);
}
