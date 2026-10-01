use super::FORMAT_VERSION;
use crate::{PbzError, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use zarrs::storage::{ReadableWritableListableStorage, StoreKey, StorePrefix};
use zarrs::{
    array::{DataType, codec::CodecChain},
    metadata::v3::ArrayMetadataV3,
};

const CONVENTION: &str = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c";
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Coordinate,
    Measurement,
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum Support {
    Supported,
    Unsupported { reason: String },
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct VariableInfo {
    pub name: String,
    pub dims: Vec<String>,
    pub shape: Vec<u64>,
    pub role: Role,
    pub metadata: Value,
    pub logical_support: Support,
}
impl VariableInfo {
    pub(super) fn attrs(&self) -> &Value {
        &self.metadata["attributes"]
    }
}
fn get_json(storage: &ReadableWritableListableStorage, key: &str) -> Result<Option<Value>> {
    let key = StoreKey::new(key).map_err(|e| PbzError::Metadata(e.to_string()))?;
    storage
        .get(&key)
        .map_err(|e| PbzError::Store(e.to_string()))?
        .map(|b| serde_json::from_slice(&b).map_err(|e| PbzError::Metadata(e.to_string())))
        .transpose()
}
fn coordinate_names(attrs: &Value, names: &mut BTreeSet<String>) {
    if let Some(value) = attrs["coordinates"].as_str() {
        names.extend(value.split_whitespace().map(str::to_owned));
    }
}
pub(super) fn encoding_limit(info: &VariableInfo) -> Option<String> {
    let dtype = info.metadata["data_type"].as_str().unwrap_or("");
    if !matches!(
        dtype,
        "bool"
            | "uint8"
            | "uint16"
            | "uint32"
            | "uint64"
            | "int8"
            | "int16"
            | "int32"
            | "int64"
            | "float32"
            | "float64"
            | "string"
    ) {
        return Some(format!("unsupported logical dtype: {dtype}"));
    }
    let attrs = info.attrs();
    for field in [
        "scale_factor",
        "add_offset",
        "calendar",
        "_Unsigned",
        "missing_value",
    ] {
        if attrs.get(field).is_some() {
            return Some(format!("unsupported logical encoding: {field}"));
        }
    }
    if attrs["units"]
        .as_str()
        .is_some_and(|s| s.contains(" since "))
    {
        return Some("calendar/time decoding is unsupported".into());
    }
    if let Some(logical) = attrs.get("dtype") {
        if logical != "bool" || dtype != "int8" {
            return Some(format!("unsupported logical dtype encoding: {logical}"));
        }
        if attrs.get("_FillValue").is_some_and(|v| !v.is_null()) {
            return Some("masked logical Boolean is unsupported".into());
        }
    }
    None
}
fn support(info: &VariableInfo) -> Support {
    let result = (|| -> std::result::Result<(), String> {
        let meta: ArrayMetadataV3 =
            serde_json::from_value(info.metadata.clone()).map_err(|e| e.to_string())?;
        DataType::from_metadata(&meta.data_type).map_err(|e| e.to_string())?;
        CodecChain::from_metadata(&meta.codecs).map_err(|e| e.to_string())?;
        if let Some(reason) = encoding_limit(info) {
            return Err(reason);
        }
        Ok(())
    })();
    match result {
        Ok(()) => Support::Supported,
        Err(reason) => Support::Unsupported { reason },
    }
}
pub(super) fn catalog(
    storage: &ReadableWritableListableStorage,
) -> Result<(Value, BTreeMap<String, VariableInfo>)> {
    let root = get_json(storage, "zarr.json")?
        .ok_or_else(|| PbzError::Metadata("R1: missing root metadata".into()))?;
    if root["zarr_format"] != 3 || root["node_type"] != "group" || !root["attributes"].is_object() {
        return Err(PbzError::Metadata("R1: expected Zarr v3 root group".into()));
    }
    let attrs = &root["attributes"];
    if !attrs["zarr_conventions"].as_array().is_some_and(|a| {
        a.iter()
            .any(|c| c["uuid"] == CONVENTION && c["name"] == "pbz:")
    }) {
        return Err(PbzError::Metadata(
            "R1: missing PBZ convention identity".into(),
        ));
    }
    let version = attrs["pbz:version"]
        .as_str()
        .ok_or_else(|| PbzError::Metadata("R1: missing version".into()))?;
    if version != FORMAT_VERSION {
        return Err(PbzError::UnsupportedFormat(version.into()));
    }
    let mut coords = BTreeSet::new();
    coordinate_names(attrs, &mut coords);
    let mut variables = BTreeMap::new();
    let children = storage
        .list_dir(&StorePrefix::root())
        .map_err(|e| PbzError::Store(e.to_string()))?;
    for prefix in children.prefixes() {
        let name = prefix.as_str().trim_end_matches('/');
        let Some(meta) = get_json(storage, &format!("{name}/zarr.json"))? else {
            continue;
        };
        if meta["node_type"] == "group" {
            continue;
        }
        if meta["node_type"] != "array" || meta["zarr_format"] != 3 {
            return Err(PbzError::Metadata(format!(
                "invalid Zarr v3 array metadata: {name}"
            )));
        }
        let shape: Vec<u64> = serde_json::from_value(meta["shape"].clone())
            .map_err(|e| PbzError::Metadata(format!("{name}: invalid shape: {e}")))?;
        let dims: Vec<String> = meta["dimension_names"]
            .as_array()
            .map(|d| {
                d.iter()
                    .map(|v| v.as_str().unwrap_or("").to_owned())
                    .collect()
            })
            .unwrap_or_default();
        if dims == [name] && shape.len() == 1 {
            coords.insert(name.to_owned());
        }
        coordinate_names(&meta["attributes"], &mut coords);
        let mut info = VariableInfo {
            name: name.into(),
            dims,
            shape,
            role: Role::Measurement,
            metadata: meta,
            logical_support: Support::Supported,
        };
        info.logical_support = support(&info);
        variables.insert(name.to_owned(), info);
    }
    for name in coords {
        if let Some(info) = variables.get_mut(&name) {
            info.role = Role::Coordinate;
        }
    }
    Ok((root, variables))
}
