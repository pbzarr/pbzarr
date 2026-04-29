mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn list_tracks() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--tracks"])
        .assert()
        .success()
        .stdout("depths\n");
}

#[test]
fn list_contigs() {
    let f = fixtures::StoreFixture::empty();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--contigs"])
        .assert()
        .success()
        .stdout(predicate::str::contains("chr1"))
        .stdout(predicate::str::contains("chr2"));
}

#[test]
fn list_columns_for_columnar_track() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 3);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--columns", "depths"])
        .assert()
        .success()
        .stdout("s0\ns1\ns2\n");
}

#[test]
fn list_columns_on_scalar_track_errors() {
    let f = fixtures::StoreFixture::with_bool_track("mask");
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--columns", "mask"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("scalar"));
}

#[test]
fn list_no_mode_flag_errors_at_parse_time() {
    let f = fixtures::StoreFixture::empty();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .assert()
        .failure();
}

#[test]
fn list_tracks_json() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--tracks", "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v, serde_json::json!(["depths"]));
}
