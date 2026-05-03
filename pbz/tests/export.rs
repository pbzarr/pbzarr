mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

#[test]
fn export_tsv_multi_column() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = f._dir.path().join("out.tsv");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .args(["--region", "chr1:0-3"])
        .assert()
        .success();
    let contents = fs::read_to_string(&out).unwrap();
    assert!(
        contents.starts_with("contig\tpos\ts0\ts1\n"),
        "got: {contents}"
    );
    assert!(contents.contains("chr1\t0\t5\t5\n"));
    assert!(contents.contains("chr1\t1\t5\t5\n"));
    assert!(contents.contains("chr1\t2\t5\t5\n"));
    // chr1\t3\t... should NOT appear because end=3 is exclusive
    assert!(!contents.contains("chr1\t3\t"));
}

#[test]
fn export_tsv_column_filter() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = f._dir.path().join("out.tsv");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .args(["--region", "chr1:0-2"])
        .args(["--column", "s1"])
        .assert()
        .success();
    let contents = fs::read_to_string(&out).unwrap();
    // Header has only the chosen column
    assert!(contents.starts_with("contig\tpos\ts1\n"), "got: {contents}");
    // Rows have just one value column
    assert!(contents.contains("chr1\t0\t5\n"));
    assert!(contents.contains("chr1\t1\t5\n"));
    // No s0 column header
    assert!(!contents.contains("\ts0"));
}

#[test]
fn export_bedgraph_collapses_runs() {
    let f = fixtures::StoreFixture::with_single_col_uint32("depths");
    let out = f._dir.path().join("out.bedgraph");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .args(["--region", "chr1:0-1000"])
        .assert()
        .success();
    let contents = fs::read_to_string(&out).unwrap();
    // All 1000 positions on chr1 have value 7 → one collapsed line.
    assert_eq!(contents, "chr1\t0\t1000\t7\n");
}

#[test]
fn export_bed_from_bool_track() {
    let f = fixtures::StoreFixture::with_bool_track("mask");
    let out = f._dir.path().join("out.bed");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("mask")
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    let contents = fs::read_to_string(&out).unwrap();
    // First chunk on chr1 (positions 0..100) is true, rest is false (filled with default).
    assert_eq!(contents, "chr1\t0\t100\n");
}

#[test]
fn export_bedgraph_rejects_multicolumn() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = f._dir.path().join("out.bedgraph");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .assert()
        .failure()
        .stderr(predicate::str::contains("single-column"));
}

#[test]
fn export_bed_rejects_non_bool() {
    let f = fixtures::StoreFixture::with_single_col_uint32("depths");
    let out = f._dir.path().join("out.bed");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .assert()
        .failure()
        .stderr(predicate::str::contains("bool"));
}

#[test]
fn export_unknown_column_errors() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = f._dir.path().join("out.tsv");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&f.path)
        .arg("depths")
        .arg("-o")
        .arg(&out)
        .args(["--column", "nonexistent"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("column not found"));
}

#[test]
fn export_bedgraph_include_zero_changes_output() {
    // Build a single-column (1D) uint32 track manually so we have known zeros.
    use ndarray::Array1;
    use pbzarr::{PbzStore, TrackConfig};
    use tempfile::TempDir;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz.zarr");
    PbzStore::create(&path, &["chr1".to_string()], &[10u64]).unwrap();
    let store = PbzStore::open(&path).unwrap();
    let cfg = TrackConfig {
        dtype: "uint32".into(),
        columns: None,
        chunk_size: 100,
        column_chunk_size: 1,
        column_dim_name: None,
        description: None,
        source: None,
        extra: serde_json::Map::new(),
    };
    let track = store.create_track("d", cfg).unwrap();
    // [0,0,5,5,0,0,7,7,0,0]
    let mut data = Array1::<u32>::zeros(10);
    data[2] = 5;
    data[3] = 5;
    data[6] = 7;
    data[7] = 7;
    track.write_chunk_1d("chr1", 0, data).unwrap();
    drop(track);
    drop(store);

    // Without --include-zero: zeros suppressed.
    let out1 = dir.path().join("a.bedgraph");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&path)
        .arg("d")
        .arg("-o")
        .arg(&out1)
        .assert()
        .success();
    let c1 = fs::read_to_string(&out1).unwrap();
    assert_eq!(c1, "chr1\t2\t4\t5\nchr1\t6\t8\t7\n");

    // With --include-zero: zeros also emitted.
    let out2 = dir.path().join("b.bedgraph");
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("export")
        .arg(&path)
        .arg("d")
        .arg("-o")
        .arg(&out2)
        .arg("--include-zero")
        .assert()
        .success();
    let c2 = fs::read_to_string(&out2).unwrap();
    assert_eq!(
        c2,
        "chr1\t0\t2\t0\nchr1\t2\t4\t5\nchr1\t4\t6\t0\nchr1\t6\t8\t7\nchr1\t8\t10\t0\n"
    );
}
