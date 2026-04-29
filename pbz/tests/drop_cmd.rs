mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn drop_with_yes_removes_track() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["drop"])
        .arg(&f.path)
        .args(["depths", "--yes"])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--tracks"])
        .assert()
        .success()
        .stdout("");
}

#[test]
fn drop_without_yes_in_non_tty_errors() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["drop"])
        .arg(&f.path)
        .arg("depths")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--yes required"));
}

#[test]
fn drop_missing_track_errors() {
    let f = fixtures::StoreFixture::empty();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["drop"])
        .arg(&f.path)
        .args(["nope", "--yes"])
        .assert()
        .failure();
}
