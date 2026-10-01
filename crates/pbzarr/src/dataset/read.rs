use super::{Dataset, VariableInfo};
use crate::{PbzError, Result};
use serde_json::Value;
use std::sync::Arc;
use zarrs::{
    array::{Array, ArraySubset},
    metadata::{ArrayMetadata, v3::ArrayMetadataV3},
};

pub(super) type ArrayHandle = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

impl Dataset {
    pub(super) fn array(&self, name: &str) -> Result<Arc<ArrayHandle>> {
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| PbzError::Store("array cache poisoned".into()))?;
        if let Some(array) = handles.get(name) {
            return Ok(array.clone());
        }
        let info = self
            .variables
            .get(name)
            .ok_or_else(|| PbzError::Metadata(format!("unknown array: {name}")))?;
        let meta: ArrayMetadataV3 = serde_json::from_value(info.metadata.clone())
            .map_err(|e| PbzError::Metadata(format!("{name}: {e}")))?;
        let array = Arc::new(
            Array::new_with_metadata(
                self.storage.clone(),
                &format!("/{name}"),
                ArrayMetadata::V3(meta),
            )
            .map_err(|e| PbzError::UnsupportedEncoding(format!("{name}: {e}")))?,
        );
        handles.insert(name.into(), array.clone());
        Ok(array)
    }
    pub(super) fn coordinate_values(&self, info: &VariableInfo) -> Result<Vec<Value>> {
        if let Some(reason) = super::metadata::encoding_limit(info) {
            return Err(PbzError::UnsupportedEncoding(format!(
                "{}: {reason}",
                info.name
            )));
        }
        let array = self.array(&info.name)?;
        let subset = ArraySubset::new_with_shape(info.shape.clone());
        macro_rules! load {
            ($t:ty) => {{
                let values: Vec<$t> = array
                    .retrieve_array_subset(&subset)
                    .map_err(|e| PbzError::Store(format!("{}: {e}", info.name)))?;
                values
                    .into_iter()
                    .map(|v| serde_json::to_value(v).unwrap_or(Value::Null))
                    .collect()
            }};
        }
        Ok(match info.metadata["data_type"].as_str().unwrap_or("") {
            "string" => load!(String),
            "uint8" => load!(u8),
            "uint16" => load!(u16),
            "uint32" => load!(u32),
            "uint64" => load!(u64),
            "int8" => load!(i8),
            "int16" => load!(i16),
            "int32" => load!(i32),
            "int64" => load!(i64),
            "float32" => load!(f32),
            "float64" => load!(f64),
            other => {
                return Err(PbzError::UnsupportedEncoding(format!(
                    "{}: unsupported coordinate type {other}",
                    info.name
                )));
            }
        })
    }
}
/// Compare a declared missing sentinel in the physical domain. Zarr fill alone
/// never marks a value as missing.
pub(super) fn missing(info: &VariableInfo, value: &Value) -> bool {
    if value.is_null() {
        return true;
    }
    let fill = &info.attrs()["_FillValue"];
    if fill.is_null() {
        return false;
    }
    if value == fill {
        return true;
    }
    // Floating coordinates may encode an integer-valued sentinel in JSON.
    if info.metadata["data_type"]
        .as_str()
        .is_some_and(|t| t.starts_with("float"))
    {
        return value.as_f64() == fill.as_f64();
    }
    let integer = |v: &Value| -> Option<i128> {
        v.as_i64()
            .map(i128::from)
            .or_else(|| v.as_u64().map(i128::from))
            .or_else(|| {
                v.as_f64()
                    .filter(|f| f.is_finite() && f.fract() == 0.0)
                    .map(|f| f as i128)
            })
    };
    integer(value).is_some_and(|v| Some(v) == integer(fill))
}
