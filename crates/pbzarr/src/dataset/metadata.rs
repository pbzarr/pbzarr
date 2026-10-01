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
    if (dtype == "bool" || attrs["dtype"] == "bool")
        && attrs.get("_FillValue").is_some_and(|v| !v.is_null())
    {
        return Some("masked logical Boolean is unsupported".into());
    }
    if matches!(dtype, "float32" | "float64")
        && let Err(e) = super::read::float_fill(info)
    {
        return Some(e.to_string());
    }
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
    let result = storage_metadata(info).and_then(|_| {
        encoding_limit(info).map_or(Ok(()), |reason| Err(PbzError::UnsupportedEncoding(reason)))
    });
    match result {
        Ok(()) => Support::Supported,
        Err(e) => Support::Unsupported {
            reason: e.to_string(),
        },
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
    if attrs["zarr_conventions"]
        .as_array()
        .is_none_or(|a| a.iter().filter(|c| c["uuid"] == CONVENTION).count() != 1)
    {
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

use super::{Dataset, GenomicAxis, Reference};
use crate::Contig;
use std::num::NonZeroU64;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Valid,
    Invalid,
    Incomplete,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationIssue {
    pub array: Option<String>,
    pub rule: String,
    pub message: String,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct ValidationReport {
    pub status: ValidationStatus,
    pub issues: Vec<ValidationIssue>,
}
impl ValidationReport {
    pub(super) fn new() -> Self {
        Self {
            status: ValidationStatus::Valid,
            issues: Vec::new(),
        }
    }
    pub(super) fn issue(
        &mut self,
        array: Option<&str>,
        rule: &str,
        message: impl Into<String>,
        incomplete: bool,
    ) {
        if !incomplete {
            self.status = ValidationStatus::Invalid;
        } else if self.status == ValidationStatus::Valid {
            self.status = ValidationStatus::Incomplete;
        }
        self.issues.push(ValidationIssue {
            array: array.map(str::to_owned),
            rule: rule.into(),
            message: message.into(),
        });
    }
    fn error(&mut self, array: Option<&str>, rule: &str, error: PbzError) -> Result<()> {
        match error {
            PbzError::UnsupportedEncoding(message) => self.issue(array, rule, message, true),
            PbzError::Metadata(message) => self.issue(array, rule, message, false),
            e => return Err(e),
        }
        Ok(())
    }
}
impl GenomicAxis {
    /// Interpret a reserved dimension name, preserving arbitrary-size decimal
    /// widths by capping arithmetic at the largest permitted contig length.
    pub fn from_dimension(name: &str) -> Result<Option<Self>> {
        if name == "position" {
            return Ok(Some(Self::Position));
        }
        let Some(width) = name.strip_prefix("bin_") else {
            return Ok(None);
        };
        if width.is_empty() || width.starts_with('0') || !width.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(PbzError::Metadata(format!(
                "G1: noncanonical genomic dimension {name}"
            )));
        }
        let bound = i64::MAX.to_string();
        let width = if width.len() > bound.len()
            || (width.len() == bound.len() && width > bound.as_str())
        {
            i64::MAX as u64
        } else {
            width
                .parse::<u64>()
                .map_err(|e| PbzError::Metadata(e.to_string()))?
        };
        Ok(Some(Self::Bin(
            NonZeroU64::new(width).expect("canonical nonzero width"),
        )))
    }
}
fn dims_valid(info: &VariableInfo) -> bool {
    let raw = info.metadata.get("dimension_names");
    let shape = &info.shape;
    let syntax = match raw {
        None => shape.is_empty(),
        Some(Value::Array(d)) => {
            d.len() == shape.len() && d.iter().all(|v| v.as_str().is_some_and(|s| !s.is_empty()))
        }
        _ => false,
    };
    syntax && info.dims.iter().collect::<BTreeSet<_>>().len() == info.dims.len()
}
impl Dataset {
    fn reference_arrays(&self) -> Result<(&VariableInfo, &VariableInfo)> {
        let r = &self.root["attributes"]["pbz:reference"];
        let resolve = |key: &str| -> Result<&VariableInfo> {
            let name = r[key]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| PbzError::Metadata(format!("R2: missing reference {key}")))?;
            zarrs::node::NodeName::new(name).map_err(|e| PbzError::Metadata(format!("R2: {e}")))?;
            self.variables
                .get(name)
                .ok_or_else(|| PbzError::Metadata(format!("R2: missing root array {name}")))
        };
        let names = resolve("names")?;
        let lengths = resolve("lengths")?;
        if names.name == lengths.name {
            return Err(PbzError::Metadata(
                "R2: reference pointers must be distinct".into(),
            ));
        }
        if names.role != Role::Coordinate || lengths.role != Role::Coordinate {
            return Err(PbzError::Metadata(
                "R3: reference arrays must be coordinates".into(),
            ));
        }
        if !dims_valid(names)
            || names.shape.len() != 1
            || names.dims != [names.name.clone()]
            || names.metadata["data_type"] != "string"
        {
            return Err(PbzError::Metadata(
                "R4: reference names must be a string dimension coordinate".into(),
            ));
        }
        if !dims_valid(lengths)
            || lengths.dims != names.dims
            || lengths.shape != names.shape
            || !matches!(
                lengths.metadata["data_type"].as_str(),
                Some(
                    "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64"
                )
            )
            || lengths.attrs()["dtype"] == "bool"
        {
            return Err(PbzError::Metadata(
                "R5: reference lengths must be integer values on the names dimension".into(),
            ));
        }
        Ok((names, lengths))
    }
    /// Load and check reference coordinates lazily; offsets retain stored order.
    pub fn reference(&self) -> Result<&Reference> {
        root_encoding(&self.root)?;
        if let Some(r) = self.reference_cache.get() {
            return Ok(r);
        }
        let (names, lengths) = self.reference_arrays()?;
        let ns = self.coordinate_values(names)?;
        let ls = self.coordinate_values(lengths)?;
        let mut seen = BTreeSet::new();
        let mut contigs = Vec::with_capacity(ns.len());
        for (name, length) in ns.iter().zip(ls.iter()) {
            let name = name
                .as_str()
                .filter(|n| !n.is_empty() && !super::read::missing(names, name))
                .ok_or_else(|| PbzError::Metadata("R4: missing or empty contig name".into()))?;
            if !seen.insert(name) {
                return Err(PbzError::Metadata("R4: duplicate contig name".into()));
            }
            if super::read::missing(lengths, length) {
                return Err(PbzError::Metadata("R6: missing reference length".into()));
            }
            let length = length
                .as_u64()
                .filter(|n| *n <= i64::MAX as u64)
                .ok_or_else(|| {
                    PbzError::Metadata(
                        "R6: reference length is outside signed 64-bit indexing".into(),
                    )
                })?;
            contigs.push(Contig {
                name: name.into(),
                length,
            });
        }
        let reference =
            Reference::new(contigs).map_err(|e| PbzError::Metadata(format!("R6: {e}")))?;
        let _ = self.reference_cache.set(reference);
        Ok(self.reference_cache.get().expect("reference initialized"))
    }
    fn validate_labels(&self, dim: &str) -> Result<()> {
        let info = self
            .variables
            .get(dim)
            .filter(|v| v.role == Role::Coordinate && v.dims == [dim] && v.shape.len() == 1)
            .ok_or_else(|| PbzError::Metadata(format!("D3: no dimension coordinate for {dim}")))?;
        if info.attrs()["dtype"] == "bool" {
            return Err(PbzError::Metadata(format!(
                "D3: Boolean labels are not string or numeric: {dim}"
            )));
        }
        let values = self.coordinate_values(info)?;
        let mut seen = BTreeSet::new();
        for value in values {
            if super::read::missing(info, &value) {
                return Err(PbzError::Metadata(format!("D3: missing label on {dim}")));
            }
            let key = value.label_key();
            if !seen.insert(key) {
                return Err(PbzError::Metadata(format!("D3: duplicate labels on {dim}")));
            }
        }
        Ok(())
    }
    /// Check metadata, reference and dimension labels without loading measurements.
    pub fn validate(&self) -> Result<ValidationReport> {
        self.validate_scope(None)
    }
    pub(super) fn validate_scope(&self, selected: Option<&str>) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        let relevant: BTreeSet<&str> = match selected {
            Some(name) => self.variables[name]
                .dims
                .iter()
                .map(String::as_str)
                .collect(),
            None => self
                .variables
                .values()
                .flat_map(|v| v.dims.iter().map(String::as_str))
                .collect(),
        };
        let reference = match self.reference() {
            Ok(r) => Some(r),
            Err(e) => {
                let rule = match &e {
                    PbzError::Metadata(m) => m
                        .split(':')
                        .next()
                        .filter(|r| r.starts_with('R'))
                        .unwrap_or("R6"),
                    _ => "R6",
                }
                .to_owned();
                report.error(None, &rule, e)?;
                None
            }
        };
        let mut extents = BTreeMap::new();
        for info in self.variables.values() {
            let included = selected.is_none()
                || selected == Some(info.name.as_str())
                || info.dims.iter().any(|d| relevant.contains(d.as_str()));
            if !included {
                continue;
            }
            if !dims_valid(info) {
                report.issue(
                    Some(&info.name),
                    "D1",
                    "axes require distinct nonempty dimension names",
                    false,
                );
                continue;
            }
            // Neighboring unsupported arrays remain inspectable. A selected
            // operation checks only its own encodings and required labels.
            if (selected.is_none() || selected == Some(info.name.as_str()))
                && let Err(e) = storage_metadata(info)
            {
                report.error(Some(&info.name), "ZARR", e)?;
            }
            if selected.is_none()
                && let Support::Unsupported { reason } = &info.logical_support
            {
                report.issue(Some(&info.name), "encoding", reason, true);
            }
            let mut genomic = Vec::new();
            for (i, (dim, &size)) in info.dims.iter().zip(&info.shape).enumerate() {
                if let Some(old) = extents.insert(dim, size)
                    && old != size
                {
                    report.issue(
                        Some(&info.name),
                        "D2",
                        format!("inconsistent extent for {dim}"),
                        false,
                    );
                }
                match GenomicAxis::from_dimension(dim) {
                    Err(e) => report.issue(Some(&info.name), "G1", e.to_string(), false),
                    Ok(Some(axis)) => {
                        genomic.push(i);
                        if self.variables.contains_key(dim) {
                            report.issue(
                                Some(dim),
                                "G4",
                                "stored genomic dimension coordinate is prohibited",
                                false,
                            );
                        }
                        if let Some(reference) = reference
                            && reference.extent(&axis)? != size
                        {
                            report.issue(
                                Some(&info.name),
                                "G2",
                                format!("wrong extent for {dim}"),
                                false,
                            );
                        }
                    }
                    Ok(None) => {}
                }
            }
            if !genomic.is_empty() && genomic != [0] {
                report.issue(
                    Some(&info.name),
                    "G3",
                    "a genomic variable must have one leading genomic axis",
                    false,
                );
            }
        }
        for dim in relevant {
            if matches!(GenomicAxis::from_dimension(dim), Ok(None))
                && let Err(e) = self.validate_labels(dim)
            {
                report.error(Some(dim), "D3", e)?;
            }
        }
        Ok(report)
    }
}

fn plugin_error(name: &str, error: zarrs::plugin::PluginCreateError) -> PbzError {
    use zarrs::plugin::PluginCreateError;
    let message = format!("{name}: {error}");
    match error {
        PluginCreateError::NameInvalid { .. } | PluginCreateError::ConfigurationInvalid(_) => {
            PbzError::Metadata(message)
        }
        _ => PbzError::UnsupportedEncoding(message),
    }
}
pub(super) fn array_error(name: &str, error: zarrs::array::ArrayCreateError) -> PbzError {
    use zarrs::array::ArrayCreateError as E;
    match error {
        E::DataTypeCreateError(e)
        | E::CodecsCreateError(e)
        | E::StorageTransformersCreateError(e)
        | E::ChunkGridCreateError(e)
        | E::ChunkKeyEncodingCreateError(e) => plugin_error(name, e),
        E::AdditionalFieldUnsupportedError(e) => {
            PbzError::UnsupportedEncoding(format!("{name}: {e}"))
        }
        E::StorageError(e) => PbzError::Store(e.to_string()),
        e => PbzError::Metadata(format!("{name}: {e}")),
    }
}
fn required_extensions(fields: &zarrs::metadata::v3::AdditionalFieldsV3) -> Result<()> {
    for (name, field) in fields {
        if field.must_understand() {
            return Err(PbzError::UnsupportedEncoding(format!(
                "required Zarr extension {name}"
            )));
        }
    }
    Ok(())
}
pub(super) fn storage_metadata(info: &VariableInfo) -> Result<ArrayMetadataV3> {
    use zarrs::array::{ChunkGrid, ChunkKeyEncoding, StorageTransformerChain};
    let meta: ArrayMetadataV3 = serde_json::from_value(info.metadata.clone())
        .map_err(|e| PbzError::Metadata(format!("{}: {e}", info.name)))?;
    required_extensions(&meta.additional_fields)?;
    let data_type =
        DataType::from_metadata(&meta.data_type).map_err(|e| plugin_error(&info.name, e))?;
    CodecChain::from_metadata(&meta.codecs).map_err(|e| plugin_error(&info.name, e))?;
    if meta.chunk_grid.name() == "regular"
        && info.metadata["chunk_grid"]["configuration"]["chunk_shape"]
            .as_array()
            .is_some_and(|s| s.len() != meta.shape.len())
    {
        return Err(PbzError::Metadata(format!(
            "{}: chunk grid rank differs from shape",
            info.name
        )));
    }
    let grid = ChunkGrid::from_metadata(&meta.chunk_grid, &meta.shape)
        .map_err(|e| plugin_error(&info.name, e))?;
    if grid.dimensionality() != meta.shape.len() {
        return Err(PbzError::Metadata(format!(
            "{}: chunk grid rank differs from shape",
            info.name
        )));
    }
    data_type
        .fill_value_v3(&meta.fill_value)
        .map_err(|e| PbzError::Metadata(format!("{}: invalid storage fill: {e}", info.name)))?;
    ChunkKeyEncoding::from_metadata(&meta.chunk_key_encoding)
        .map_err(|e| plugin_error(&info.name, e))?;
    let path = zarrs::node::NodePath::new(&format!("/{}", info.name))
        .map_err(|e| PbzError::Metadata(e.to_string()))?;
    StorageTransformerChain::from_metadata(&meta.storage_transformers, &path)
        .map_err(|e| plugin_error(&info.name, e))?;
    Ok(meta)
}
fn root_encoding(root: &Value) -> Result<()> {
    let meta: zarrs::metadata::v3::GroupMetadataV3 =
        serde_json::from_value(root.clone()).map_err(|e| PbzError::Metadata(format!("R1: {e}")))?;
    required_extensions(&meta.additional_fields)
}
