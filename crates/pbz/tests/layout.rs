//! The chunk and shard layout of the tracks that `pbz import` creates.

use std::ffi::OsString;
use std::path::Path;

use serde_json::{Value, json};
use tempfile::TempDir;

mod common;
use common::{run_pbz, stdout_of};

fn source(name: &str, label: &str) -> OsString {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name);
    format!("{}:{label}", path.display()).into()
}

fn cohort() -> Vec<OsString> {
    vec![
        source("cohort_s0.d4", "s0"),
        source("cohort_s1.d4", "s1"),
        source("cohort_s2.d4", "s2"),
    ]
}

fn import_d4(store: &Path, flags: &[&str], sources: Vec<OsString>) {
    let mut args: Vec<OsString> = vec![
        "import".into(),
        "d4".into(),
        store.into(),
        "--track".into(),
        "depth".into(),
        "--no-progress".into(),
    ];
    args.extend(flags.iter().map(OsString::from));
    args.extend(sources);
    stdout_of(&run_pbz(args));
}

/// The `depth` metadata and its position extent.
fn depth(store: &Path) -> (Value, u64) {
    let dataset = pbzarr::Dataset::open_read_only(store).unwrap();
    let info = &dataset.variables()["depth"];
    (info.metadata.clone(), info.shape[0])
}

/// The default shard: 1,024 inner chunks, capped at the extent rounded up
/// to whole chunks.
fn default_shard(extent: u64) -> u64 {
    (65_536 * 1_024).min(extent.div_ceil(65_536) * 65_536)
}

fn assert_layout(meta: &Value, outer: Value, inner: Value) {
    assert_eq!(meta["chunk_grid"]["configuration"]["chunk_shape"], outer);
    let sharding = &meta["codecs"][0];
    assert_eq!(sharding["name"], "sharding_indexed");
    assert_eq!(sharding["configuration"]["chunk_shape"], inner);
}

#[test]
fn a_default_cohort_import_has_one_column_per_chunk_and_blosc_zstd5() {
    let dir = TempDir::new().unwrap();
    for store in ["cohort.pbz", "cohort.pbz.zip"] {
        let store = dir.path().join(store);
        import_d4(&store, &[], cohort());
        let (meta, extent) = depth(&store);
        assert_layout(&meta, json!([default_shard(extent), 3]), json!([65_536, 1]));
        let blosc = &meta["codecs"][0]["configuration"]["codecs"][1];
        assert_eq!(blosc["name"], "blosc", "{store:?}");
        assert_eq!(blosc["configuration"]["cname"], "zstd");
        assert_eq!(blosc["configuration"]["clevel"], 5);
        assert_eq!(blosc["configuration"]["shuffle"], "shuffle");
    }
}

#[test]
fn a_default_single_source_import_is_rank_one() {
    let dir = TempDir::new().unwrap();
    for store in ["single.pbz", "single.pbz.zip"] {
        let store = dir.path().join(store);
        import_d4(&store, &[], vec![source("cohort_s0.d4", "s0")]);
        let (meta, extent) = depth(&store);
        assert_eq!(meta["shape"], json!([extent]), "{store:?}");
        assert_layout(&meta, json!([default_shard(extent)]), json!([65_536]));
    }
}

#[test]
fn layout_flags_set_the_chunk_and_shard() {
    let dir = TempDir::new().unwrap();
    let store = dir.path().join("flags.pbz");
    let flags = ["--chunk", "100", "--shard", "4", "--shard-columns", "2"];
    import_d4(&store, &flags, cohort());
    let (meta, _) = depth(&store);
    assert_layout(&meta, json!([400, 2]), json!([100, 1]));
}
