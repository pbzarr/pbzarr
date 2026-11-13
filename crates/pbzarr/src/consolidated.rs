//! Root consolidated metadata in the zarr-python v3 inline form.

use std::io::Write;
use std::path::Path;

use serde_json::{Value, json};
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage, StoreKey,
    StorePrefix, WritableStorageTraits,
};

use crate::{PbzError, Result};

/// Set `root["consolidated_metadata"]` to the `zarr.json` of every node under
/// the store root, inlined verbatim and keyed by store-relative path. A stale
/// copy is replaced wholesale.
pub(crate) fn insert_snapshot(
    storage: &ReadableWritableListableStorage,
    root: &mut Value,
) -> Result<()> {
    let mut nodes = Vec::new();
    collect_nodes(storage, &StorePrefix::root(), "", &mut nodes)?;
    // zarr-python orders the flat map by depth, then case-folded path.
    nodes.sort_by(|(a, _), (b, _)| {
        a.matches('/')
            .count()
            .cmp(&b.matches('/').count())
            .then_with(|| a.to_lowercase().cmp(&b.to_lowercase()))
    });
    let metadata: serde_json::Map<String, Value> = nodes.into_iter().collect();
    root.as_object_mut()
        .ok_or_else(|| PbzError::Metadata("store root zarr.json is not an object".into()))?
        .insert(
            "consolidated_metadata".to_owned(),
            consolidated_field(metadata),
        );
    Ok(())
}

/// Replace the root `zarr.json` with `root`. Under a filesystem `dir` the
/// document goes to a temporary file beside it that is then renamed over it,
/// so a reader never sees a partial root.
pub(crate) fn replace_root(
    storage: &ReadableWritableListableStorage,
    dir: Option<&Path>,
    root: &Value,
) -> Result<()> {
    let encoded = serde_json::to_vec_pretty(root)
        .map_err(|e| PbzError::Metadata(format!("store root: encode zarr.json: {e}")))?;
    let Some(dir) = dir else {
        return storage
            .set(&zarr_json_key("")?, encoded.into())
            .map_err(|e| PbzError::Store(e.to_string()));
    };
    let io = |e: std::io::Error| PbzError::Store(format!("store root: replace zarr.json: {e}"));
    let temporary = dir.join(format!(".zarr.json.{}.tmp", std::process::id()));
    let mut file = std::fs::File::create(&temporary).map_err(io)?;
    file.write_all(&encoded).map_err(io)?;
    file.sync_all().map_err(io)?;
    drop(file);
    std::fs::rename(&temporary, dir.join("zarr.json")).map_err(|e| {
        let _ = std::fs::remove_file(&temporary);
        io(e)
    })
}

/// The `zarr.json` store key of the node at store-relative `path` (empty for
/// the store root).
fn zarr_json_key(path: &str) -> Result<StoreKey> {
    let key = if path.is_empty() {
        "zarr.json".to_owned()
    } else {
        format!("{path}/zarr.json")
    };
    StoreKey::new(key).map_err(|e| PbzError::Metadata(e.to_string()))
}

/// A `consolidated_metadata` field value wrapping `metadata`.
fn consolidated_field(metadata: serde_json::Map<String, Value>) -> Value {
    json!({
        "kind": "inline",
        "must_understand": false,
        "metadata": metadata,
    })
}

/// Collect `(store-relative path, zarr.json document)` for every node below
/// `prefix`, depth first.
///
/// Arrays terminate the walk: everything under an array prefix is chunk data,
/// not nodes. A directory without a `zarr.json` is not a node (leftover junk,
/// or a chunk directory) and is skipped without descending.
fn collect_nodes(
    storage: &ReadableWritableListableStorage,
    prefix: &StorePrefix,
    path: &str,
    out: &mut Vec<(String, Value)>,
) -> Result<()> {
    let listing = storage
        .list_dir(prefix)
        .map_err(|e| PbzError::Store(e.to_string()))?;
    for child in listing.prefixes() {
        let name = child.as_str().trim_end_matches('/').rsplit('/').next();
        let Some(name) = name.filter(|n| !n.is_empty()) else {
            continue;
        };
        let node_path = if path.is_empty() {
            name.to_owned()
        } else {
            format!("{path}/{name}")
        };
        let Some(bytes) = storage
            .get(&zarr_json_key(&node_path)?)
            .map_err(|e| PbzError::Store(e.to_string()))?
        else {
            continue;
        };
        let mut doc: Value = serde_json::from_slice(&bytes).map_err(|e| {
            PbzError::Metadata(format!("node '{node_path}': invalid zarr.json: {e}"))
        })?;
        let obj = doc.as_object_mut().ok_or_else(|| {
            PbzError::Metadata(format!("node '{node_path}': zarr.json is not an object"))
        })?;
        let is_group = obj.get("node_type").and_then(Value::as_str) == Some("group");
        if is_group {
            // zarr-python stamps an empty `consolidated_metadata` on every
            // group entry of the flat map (the nesting is rebuilt from the
            // paths on read), so a childless group never sends a reader back
            // to the store to list it. Match that.
            obj.insert(
                "consolidated_metadata".to_owned(),
                consolidated_field(serde_json::Map::new()),
            );
        }
        out.push((node_path.clone(), doc));
        if is_group {
            let child_prefix = StorePrefix::new(format!("{node_path}/"))
                .map_err(|e| PbzError::Metadata(e.to_string()))?;
            collect_nodes(storage, &child_prefix, &node_path, out)?;
        }
    }
    Ok(())
}
