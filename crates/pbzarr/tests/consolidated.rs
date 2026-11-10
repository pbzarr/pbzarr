//! Store-root consolidated metadata (zarr-python v3 flavor).

use std::path::{Path, PathBuf};
use std::process::Command;

use pbzarr::Contig;
use pbzarr::TrackConfig;
use pbzarr::import::Import;
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Dataset, Reference};
use serde_json::Value;
use tempfile::TempDir;

/// Every node the fixture store is expected to publish, store-relative.
const EXPECTED_NODES: &[&str] = &["af", "contig_id", "contig_length", "depth", "sample"];

#[derive(Clone)]
struct OneReader {
    genome: Reference,
    schema: OutputSchema,
}

impl ValueReader for OneReader {
    fn contigs(&self) -> &Reference {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        _contig: &str,
        _start: u64,
        _end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        match &mut outputs[0] {
            OutputSinkMut::I32(dst) => dst.fill(1),
            OutputSinkMut::F32(dst) => dst.fill(1.0),
            _ => panic!("test reader supports i32/f32 sinks only"),
        }
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(self.clone())
    }
}

fn import(ds: &mut Dataset, name: &str, cfg: TrackConfig) {
    let reader = OneReader {
        genome: ds.reference().unwrap().clone(),
        schema: OutputSchema::single("value", cfg.dtype),
    };
    let width = cfg.columns.as_ref().map_or(1, Vec::len);
    let builder = Import::from_readers(vec![reader; width])
        .unwrap()
        .into_dataset(ds, vec![(name.into(), cfg.clone())]);
    let builder = if cfg.columns.is_some() {
        builder.readers_as_columns()
    } else {
        builder
    };
    builder.run().unwrap();
}

/// A store with one 2-D cohort track and one 1-D track.
fn fixture(path: &Path) -> Dataset {
    let reference = Reference::new(vec![
        Contig {
            name: "chr1".into(),
            length: 10,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap();
    let mut ds = Dataset::create(path, reference).unwrap();
    import(
        &mut ds,
        "af",
        TrackConfig::new(Dtype::F32)
            .columns(vec!["s1".into(), "s2".into()])
            .column_dim("sample"),
    );
    import(&mut ds, "depth", TrackConfig::new(Dtype::I32));
    ds
}

fn root_metadata(path: &Path) -> Value {
    let bytes = std::fs::read(path.join("zarr.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn import_consolidates_every_node_into_the_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    fixture(&path);

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
            .any(|c| c["name"] == pbzarr::PBZ_CONVENTION_NAME),
        "root zarr_conventions preserved: {root}"
    );

    let metadata = consolidated["metadata"].as_object().unwrap();
    let mut got: Vec<&str> = metadata.keys().map(String::as_str).collect();
    got.sort_unstable();
    assert_eq!(got, EXPECTED_NODES);

    // Each value is a full zarr.json document, not a stub.
    let af = &metadata["af"];
    assert_eq!(af["node_type"], "array");
    assert_eq!(af["zarr_format"], 3);
    assert_eq!(af["shape"], serde_json::json!([17, 2]));
    assert_eq!(af["data_type"], "float32");
    for field in [
        "chunk_grid",
        "chunk_key_encoding",
        "codecs",
        "dimension_names",
    ] {
        assert!(af.get(field).is_some(), "af missing {field}");
    }
    assert_eq!(metadata["depth"]["shape"], serde_json::json!([17]));
    assert_eq!(metadata["depth"]["data_type"], "int32");
    assert!(af.get("consolidated_metadata").is_none());

    // The consolidated copy is advisory: per-node documents are untouched.
    let track: Value =
        serde_json::from_slice(&std::fs::read(path.join("af/zarr.json")).unwrap()).unwrap();
    assert!(track.get("consolidated_metadata").is_none());
}

/// zarr-python resolves only from the root map when it is present, so a
/// track that a later import adds must be in the map.
#[test]
fn a_later_import_refreshes_the_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    fixture(&path);

    let mut ds = Dataset::open(&path).unwrap();
    import(&mut ds, "extra", TrackConfig::new(Dtype::I32));

    let root = root_metadata(&path);
    let metadata = root["consolidated_metadata"]["metadata"]
        .as_object()
        .unwrap();
    assert_eq!(metadata["extra"]["shape"], serde_json::json!([17]));
    for node in EXPECTED_NODES {
        assert!(metadata.contains_key(*node), "{node} lost on refresh");
    }
    assert_eq!(metadata.len(), EXPECTED_NODES.len() + 1);

    let expected: Vec<String> = EXPECTED_NODES
        .iter()
        .map(|s| (*s).to_owned())
        .chain(["extra".into()])
        .collect();
    run_python_validator(
        &path,
        &expected,
        "a_later_import_refreshes_the_root_metadata",
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
