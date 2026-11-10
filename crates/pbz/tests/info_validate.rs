use tempfile::TempDir;

mod common;
use common::{build_store, run_pbz, stdout_of};

#[test]
fn info_json_lists_tracks_and_dtypes() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["info".into(), store.into_os_string(), "--json".into()]);
    let report: serde_json::Value = serde_json::from_str(&stdout_of(&output)).unwrap();
    assert_eq!(report["format_version"], "0.5");
    assert_eq!(report["dimensions"]["position"], 16);
    let variables = &report["variables"];
    assert_eq!(variables["depth"]["role"], "track");
    assert_eq!(variables["depth"]["dtype"], "int32");
    assert_eq!(variables["af"]["dtype"], "float32");
    assert_eq!(
        variables["af"]["dims"],
        serde_json::json!(["position", "sample"])
    );
    assert_eq!(variables["af"]["shards"], serde_json::json!([8, 1]));
    assert!(variables["depth"]["objects"].as_u64().unwrap() > 0);
}

#[test]
fn validate_exits_one_when_the_reference_is_broken() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["validate".into(), store.clone().into_os_string()]);
    assert!(stdout_of(&output).contains("valid"));

    std::fs::remove_dir_all(store.join("contig_length")).unwrap();
    let output = run_pbz(["validate".into(), store.into_os_string()]);
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("not valid"), "{text}");
    assert!(text.contains("missing root array contig_length"), "{text}");
    assert!(!text.contains("R2"), "{text}");
}
