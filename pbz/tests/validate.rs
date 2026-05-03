mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

#[test]
fn validate_clean_store_succeeds() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["validate"])
        .arg(&f.path)
        .assert()
        .success();
}

#[test]
fn validate_missing_root_attr_fails() {
    let f = fixtures::StoreFixture::empty();
    let zarr_json = f.path.join("zarr.json");
    fs::write(&zarr_json, r#"{"node_type":"group","zarr_format":3}"#).unwrap();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["validate"])
        .arg(&f.path)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("missing_root_attr"));
}

#[test]
fn validate_json_emits_findings_array() {
    let f = fixtures::StoreFixture::empty();
    let zarr_json = f.path.join("zarr.json");
    fs::write(&zarr_json, r#"{"node_type":"group","zarr_format":3}"#).unwrap();
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .args(["validate"])
        .arg(&f.path)
        .args(["--json"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["findings"].is_array());
    assert!(
        v["findings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["code"] == "missing_root_attr")
    );
}

#[test]
fn validate_does_not_crash_on_malformed_track_zarr_json() {
    let f = fixtures::StoreFixture::with_single_col_uint32("depths");
    // corrupt the track group's zarr.json with non-JSON content
    let track_zarr_json = f.path.join("tracks").join("depths").join("zarr.json");
    std::fs::write(&track_zarr_json, "this is not json").unwrap();
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .args(["validate"])
        .arg(&f.path)
        .args(["--json"])
        .output()
        .unwrap();
    // Must exit with code 2 and emit valid JSON that includes track_zarr_json_malformed
    assert_eq!(out.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let codes: Vec<&str> = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|f| f["code"].as_str())
        .collect();
    assert!(
        codes
            .iter()
            .any(|c| *c == "track_zarr_json_malformed" || *c == "track_open_failed"),
        "expected a track_zarr_json_malformed or track_open_failed finding, got: {codes:?}"
    );
}
