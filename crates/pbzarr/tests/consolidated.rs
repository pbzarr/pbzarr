//! Store-root consolidated metadata (zarr-python v3 flavor).

use std::path::{Path, PathBuf};
use std::process::Command;

use ndarray::{Array1, Array2};
use pbzarr::genome::Contig;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore, ScaleConfig, TrackConfig, scale};
use serde_json::Value;
use tempfile::TempDir;

/// Every node a scaled fixture store is expected to publish, store-relative.
const EXPECTED_NODES: &[&str] = &[
    "af",
    "af/contigs",
    "af/offsets",
    "af/sample",
    "af/values",
    "af/scales",
    "af/scales/4",
    "af/scales/4/mean",
    "af/scales/8",
    "af/scales/8/mean",
    "depth",
    "depth/contigs",
    "depth/offsets",
    "depth/values",
];

/// A store with one 2-D cohort track (scaled at factors 4 and 8) and one
/// unscaled 1-D track, so consolidation is exercised across node kinds and
/// across tracks that `scale` never touched.
fn scaled_fixture(path: &Path) {
    let genome = || {
        Genome::new(vec![
            Contig {
                name: "chr1".into(),
                length: 10,
            },
            Contig {
                name: "chr2".into(),
                length: 7,
            },
        ])
        .unwrap()
    };

    let mut store = PbzStore::create(path).unwrap();
    store
        .create_track(
            "af",
            genome(),
            TrackConfig::new(Dtype::F32)
                .columns(vec!["s1".into(), "s2".into()])
                .column_dim("sample"),
        )
        .unwrap();
    store
        .create_track("depth", genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let af = store.track("af").unwrap();
    let region = af.genome().resolve(&"chr1:0-10".parse().unwrap()).unwrap();
    let data: Vec<f32> = (0..20).map(|v| v as f32).collect();
    af.write_region(
        &region,
        Array2::from_shape_vec((10, 2), data).unwrap().into_dyn(),
    )
    .unwrap();

    let depth = store.track("depth").unwrap();
    let region = depth
        .genome()
        .resolve(&"chr1:0-10".parse().unwrap())
        .unwrap();
    depth
        .write_region(
            &region,
            Array1::from((0..10).collect::<Vec<i32>>()).into_dyn(),
        )
        .unwrap();

    scale(
        &store,
        "af",
        &ScaleConfig {
            factors: Some(vec![4, 8]),
            ..ScaleConfig::default()
        },
    )
    .unwrap();
}

fn root_metadata(path: &Path) -> Value {
    let bytes = std::fs::read(path.join("zarr.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn scale_consolidates_every_node_into_the_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    scaled_fixture(&path);

    let root = root_metadata(&path);
    let consolidated = &root["consolidated_metadata"];
    assert_eq!(consolidated["kind"], "inline");
    assert_eq!(consolidated["must_understand"], Value::Bool(false));

    // The root's own content survives the rewrite.
    assert_eq!(root["node_type"], "group");
    assert_eq!(root["zarr_format"], 3);
    assert!(
        root["attributes"]["zarr_conventions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["name"] == "perbase"),
        "root zarr_conventions preserved: {root}"
    );

    let metadata = consolidated["metadata"].as_object().unwrap();
    let mut got: Vec<&str> = metadata.keys().map(String::as_str).collect();
    got.sort_unstable();
    let mut want = EXPECTED_NODES.to_vec();
    want.sort_unstable();
    assert_eq!(got, want);

    // Each value is a full zarr.json document, not a stub.
    assert_eq!(metadata["af"]["node_type"], "group");
    assert_eq!(
        metadata["af"]["attributes"]["multiscales"]["layout"][1]["asset"],
        "scales/4/mean"
    );
    assert_eq!(metadata["af/scales"]["node_type"], "group");

    let values = &metadata["af/values"];
    assert_eq!(values["node_type"], "array");
    assert_eq!(values["zarr_format"], 3);
    assert_eq!(values["shape"], serde_json::json!([17, 2]));
    assert_eq!(values["data_type"], "float32");
    for field in [
        "chunk_grid",
        "chunk_key_encoding",
        "codecs",
        "dimension_names",
    ] {
        assert!(values.get(field).is_some(), "af/values missing {field}");
    }

    // Level arrays: ceil(10/4)+ceil(7/4) = 5 bins, ceil(10/8)+ceil(7/8) = 3.
    let level4 = &metadata["af/scales/4/mean"];
    assert_eq!(level4["node_type"], "array");
    assert_eq!(level4["shape"], serde_json::json!([5, 2]));
    assert_eq!(level4["data_type"], "float32");
    assert_eq!(
        metadata["af/scales/8/mean"]["shape"],
        serde_json::json!([3, 2])
    );

    // Untouched tracks are consolidated too.
    assert_eq!(metadata["depth/values"]["shape"], serde_json::json!([17]));
    assert_eq!(metadata["depth/values"]["data_type"], "int32");

    // Group entries carry the empty nested field zarr-python writes; array
    // entries never do.
    assert_eq!(
        metadata["af/scales/4"]["consolidated_metadata"],
        serde_json::json!({"kind": "inline", "must_understand": false, "metadata": {}})
    );
    assert!(metadata["af/values"].get("consolidated_metadata").is_none());

    // The consolidated copy is advisory: per-node documents are untouched.
    let track: Value =
        serde_json::from_slice(&std::fs::read(path.join("af/zarr.json")).unwrap()).unwrap();
    assert!(track.get("consolidated_metadata").is_none());
}

#[test]
fn consolidate_metadata_is_idempotent() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    scaled_fixture(&path);
    let first = root_metadata(&path);

    let store = PbzStore::open(&path).unwrap();
    store.consolidate_metadata().unwrap();
    let second = root_metadata(&path);
    assert_eq!(first, second, "re-consolidation is a no-op");
}

/// Track completion is a publication event, so a track created *after* a
/// pyramid publish must not be left out of the root map: zarr-python resolves
/// solely from the map when it is present, which would make the new track
/// silently invisible.
#[test]
fn track_creation_after_scale_refreshes_the_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    scaled_fixture(&path);

    let mut store = PbzStore::open(&path).unwrap();
    store
        .create_track(
            "extra",
            Genome::new(vec![Contig {
                name: "chr1".into(),
                length: 4,
            }])
            .unwrap(),
            TrackConfig::new(Dtype::I32),
        )
        .unwrap();
    // No explicit consolidate_metadata() call: create_track must refresh it.

    let root = root_metadata(&path);
    let metadata = root["consolidated_metadata"]["metadata"]
        .as_object()
        .unwrap();
    for node in ["extra", "extra/values", "extra/contigs", "extra/offsets"] {
        assert!(
            metadata.contains_key(node),
            "{node} missing from {metadata:?}"
        );
    }
    assert_eq!(metadata["extra/values"]["shape"], serde_json::json!([4]));
    // The scaled track survives the refresh, and the previous consolidated
    // field was replaced rather than nested.
    for node in EXPECTED_NODES {
        assert!(metadata.contains_key(*node), "{node} lost on refresh");
    }
    assert!(
        metadata["extra"]["consolidated_metadata"]["metadata"]
            .as_object()
            .unwrap()
            .is_empty()
    );

    // ...and zarr-python still resolves the whole store from the root map.
    let expected: Vec<String> = EXPECTED_NODES
        .iter()
        .map(|s| (*s).to_owned())
        .chain(["extra".into(), "extra/values".into()])
        .collect();
    run_python_validator(
        &path,
        &expected,
        "track_creation_after_scale_refreshes_the_root_metadata",
    );
}

/// A crash between the pyramid publication write and the consolidation
/// refresh leaves the pyramid on disk but hidden from the root map. Retrying
/// `scale` lands in the already-published branch: it must still refuse the
/// rescale, but heal the root map on the way out, or the store stays stale
/// until some unrelated publication event.
#[test]
fn rescale_refusal_still_heals_a_stale_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    scaled_fixture(&path);

    // Simulate the crash: the pyramid is published, the root map is not.
    let mut root = root_metadata(&path);
    root.as_object_mut()
        .unwrap()
        .remove("consolidated_metadata");
    std::fs::write(path.join("zarr.json"), serde_json::to_vec(&root).unwrap()).unwrap();
    assert!(root_metadata(&path).get("consolidated_metadata").is_none());

    let store = PbzStore::open(&path).unwrap();
    let err = scale(
        &store,
        "af",
        &ScaleConfig {
            factors: Some(vec![4, 8]),
            ..ScaleConfig::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("unpublish/rescale"), "{err}");

    // The refusal refreshed the root map anyway: the published levels are
    // visible again to a reader resolving solely from it.
    let metadata = root_metadata(&path)["consolidated_metadata"]["metadata"]
        .as_object()
        .unwrap()
        .clone();
    for node in EXPECTED_NODES {
        assert!(metadata.contains_key(*node), "{node} missing after heal");
    }
    assert_eq!(
        metadata["af/scales/4/mean"]["shape"],
        serde_json::json!([5, 2])
    );
}

/// The repository root (the pixi workspace), from this crate's manifest dir.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap()
}

/// True when `pixi run python` can import zarr in this checkout.
fn pixi_zarr_available(root: &Path) -> bool {
    Command::new("pixi")
        .current_dir(root)
        .args(["run", "python", "-c", "import zarr"])
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Assert released zarr-python resolves `expected` from `store`'s consolidated
/// metadata alone. Skips (loudly) when pixi or zarr is unavailable.
fn run_python_validator(store: &Path, expected: &[String], test: &str) {
    let root = workspace_root();
    if !pixi_zarr_available(&root) {
        eprintln!("skip {test} python validation: pixi/zarr unavailable");
        return;
    }
    let out = Command::new("pixi")
        .current_dir(&root)
        .args(["run", "python", "scripts/validate_consolidated.py"])
        .arg(store)
        .args(expected)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "validate_consolidated.py failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
