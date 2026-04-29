mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn cat_default_emits_tsv() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .arg("cat")
        .arg(&f.path)
        .arg("depths")
        .args(["--region", "chr1:0-2"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    assert!(s.starts_with("contig\tpos\ts0\ts1\n"), "got: {s}");
    assert!(s.contains("chr1\t0\t5\t5\n"));
    assert!(s.contains("chr1\t1\t5\t5\n"));
    assert!(!s.contains("chr1\t2\t"));
}

#[test]
fn cat_bedgraph_to_stdout() {
    let f = fixtures::StoreFixture::with_single_col_uint32("depths");
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .arg("cat")
        .arg(&f.path)
        .arg("depths")
        .args(["--region", "chr1:0-1000"])
        .args(["--format", "bedgraph"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    assert_eq!(s, "chr1\t0\t1000\t7\n");
}

#[test]
fn cat_region_clip() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("cat")
        .arg(&f.path)
        .arg("depths")
        .args(["--region", "chr2:0-3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("chr2\t0\t5\t5\n"))
        .stdout(predicate::str::contains("chr2\t2\t5\t5\n"))
        .stdout(predicate::str::contains("chr2\t3\t").not());
}
