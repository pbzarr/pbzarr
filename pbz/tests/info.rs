mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn info_human_lists_contigs_and_tracks() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .arg("info")
        .arg(&f.path)
        .assert()
        .success()
        .stdout(predicate::str::contains("perbase_zarr version: 0.1"))
        .stdout(predicate::str::contains("chr1\t1000"))
        .stdout(predicate::str::contains("chr2\t500"))
        .stdout(predicate::str::contains("track: depths"))
        .stdout(predicate::str::contains("dtype: uint32"));
}

#[test]
fn info_json_is_valid_json() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 5, 2);
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .arg("info")
        .arg(&f.path)
        .arg("--json")
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["version"], "0.1");
    assert!(v["contigs"].is_array());
    assert!(v["tracks"].is_array());
}
