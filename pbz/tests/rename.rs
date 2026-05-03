mod fixtures;

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn rename_track_renames() {
    let f = fixtures::StoreFixture::with_uint32_track("depths", 1, 2);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["rename"])
        .arg(&f.path)
        .args(["depths", "coverage"])
        .assert()
        .success();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&f.path)
        .args(["--tracks"])
        .assert()
        .success()
        .stdout("coverage\n");
}

#[test]
fn rename_track_old_missing() {
    let f = fixtures::StoreFixture::empty();
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["rename"])
        .arg(&f.path)
        .args(["nope", "x"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found"));
}

#[test]
fn rename_track_new_exists() {
    let f = fixtures::StoreFixture::with_uint32_track("a", 1, 2);
    let store = pbzarr::PbzStore::open(&f.path).unwrap();
    let cfg = pbzarr::TrackConfig {
        dtype: "uint32".into(),
        columns: None,
        chunk_size: 100,
        column_chunk_size: 1,
        column_dim_name: None,
        description: None,
        source: None,
        extra: serde_json::Map::new(),
    };
    store.create_track("b", cfg).unwrap();
    drop(store);
    Command::cargo_bin("pbz")
        .unwrap()
        .args(["rename"])
        .arg(&f.path)
        .args(["a", "b"])
        .assert()
        .failure();
}
