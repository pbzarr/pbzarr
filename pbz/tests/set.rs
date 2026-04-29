mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn set_description_persists() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["set"])
        .arg(&f.path)
        .args(["depths", "--description", "hello"])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["info"])
        .arg(&f.path)
        .assert()
        .success()
        .stdout(predicate::str::contains("description: hello"));
}

#[test]
fn set_source_persists() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["set"])
        .arg(&f.path)
        .args(["depths", "--source", "mosdepth"])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["info"])
        .arg(&f.path)
        .assert()
        .success()
        .stdout(predicate::str::contains("source: mosdepth"));
}

#[test]
fn set_empty_description_clears() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["set"])
        .arg(&f.path)
        .args(["depths", "--description", "first"])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["set"])
        .arg(&f.path)
        .args(["depths", "--description", ""])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["info"])
        .arg(&f.path)
        .assert()
        .success()
        .stdout(predicate::str::contains("description").not());
}

#[test]
fn set_no_field_errors_at_parse_time() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["set"])
        .arg(&f.path)
        .arg("depths")
        .assert()
        .failure();
}
