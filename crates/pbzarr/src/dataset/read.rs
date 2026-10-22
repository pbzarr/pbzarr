use super::ValidationStatus;
use super::{Dataset, VariableInfo};
use crate::{PbzError, Result};
use base64::Engine;
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;
use std::sync::Arc;
use zarrs::{
    array::{Array, ArrayShardedExt as _, ArraySubset},
    metadata::ArrayMetadata,
};

pub(super) type ArrayHandle = Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>;

/// Keep coordinate meaning in its physical domain, including infinities.
pub(super) enum CoordinateValue {
    String(String),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}
impl CoordinateValue {
    pub(super) fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self {
            Some(v)
        } else {
            None
        }
    }
    pub(super) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(v) => Some(*v),
            Self::Signed(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    pub(super) fn label_key(&self) -> String {
        match self {
            Self::String(v) => format!("s:{v}"),
            Self::Signed(v) => format!("i:{v}"),
            Self::Unsigned(v) => format!("u:{v}"),
            Self::Float(v) => format!("f:{}", if *v == 0.0 { 0 } else { v.to_bits() }),
        }
    }
}

/// Guard both ndarray's addressable shape and zarrs' byte allocation arithmetic.
fn checked_shape<T>(shape: &[u64]) -> Result<Vec<usize>> {
    let limit = || {
        PbzError::UnsupportedEncoding("requested buffer exceeds addressable allocation size".into())
    };
    let dims: Vec<usize> = shape
        .iter()
        .map(|&n| {
            usize::try_from(n)
                .ok()
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(limit)
        })
        .collect::<Result<_>>()?;
    let count = if dims.contains(&0) {
        0
    } else {
        dims.iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .ok_or_else(limit)?
    };
    if count > isize::MAX as usize / std::mem::size_of::<T>().max(1) {
        return Err(limit());
    }
    Ok(dims)
}

pub(super) fn float_fill(info: &VariableInfo) -> Result<Option<f64>> {
    let fill = &info.attrs()["_FillValue"];
    if fill.is_null() {
        return Ok(None);
    }
    if let Some(number) = fill.as_f64() {
        return Ok(Some(number));
    }
    if let Some(encoded) = fill.as_str() {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|e| {
                PbzError::UnsupportedEncoding(format!(
                    "{}: invalid floating _FillValue: {e}",
                    info.name
                ))
            })?;
        let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
            PbzError::UnsupportedEncoding(format!(
                "{}: floating _FillValue must encode a little-endian double",
                info.name
            ))
        })?;
        return Ok(Some(f64::from_le_bytes(bytes)));
    }
    Err(PbzError::UnsupportedEncoding(format!(
        "{}: unsupported floating _FillValue",
        info.name
    )))
}

impl Dataset {
    /// Decode unit of `name`: the subchunk when sharded, else the chunk (also
    /// when a codec before sharding hides the subchunk shape).
    pub(crate) fn inner_chunk_shape(&self, name: &str) -> Result<Vec<u64>> {
        let array = self.array(name)?;
        if let Some(sub) = array.effective_subchunk_shape() {
            return Ok(sub.iter().map(|n| n.get()).collect());
        }
        let origin = vec![0; array.dimensionality()];
        let chunk = array
            .chunk_shape(&origin)
            .map_err(|e| PbzError::Store(format!("chunk shape for {name}: {e}")))?;
        Ok(chunk.iter().map(|n| n.get()).collect())
    }

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
        let meta = super::metadata::storage_metadata(info)?;
        let array = Arc::new(
            Array::new_with_metadata(
                self.storage.clone(),
                &format!("/{name}"),
                ArrayMetadata::V3(meta),
            )
            .map_err(|e| super::metadata::array_error(name, e))?,
        );
        handles.insert(name.into(), array.clone());
        Ok(array)
    }
    pub(super) fn coordinate_values(&self, info: &VariableInfo) -> Result<Vec<CoordinateValue>> {
        if let Some(reason) = super::metadata::encoding_limit(info) {
            return Err(PbzError::UnsupportedEncoding(format!(
                "{}: {reason}",
                info.name
            )));
        }
        let array = self.array(&info.name)?;
        let subset = ArraySubset::new_with_shape(info.shape.clone());
        macro_rules! load {
            ($t:ty,$convert:expr) => {{
                checked_shape::<$t>(&info.shape)?;
                // The interpretation buffer must be addressable as well.
                checked_shape::<CoordinateValue>(&info.shape)?;
                let values: Vec<$t> = array
                    .retrieve_array_subset(&subset)
                    .map_err(|e| PbzError::Store(format!("{}: {e}", info.name)))?;
                values.into_iter().map($convert).collect()
            }};
        }
        Ok(match info.metadata["data_type"].as_str().unwrap_or("") {
            "string" => load!(String, CoordinateValue::String),
            "uint8" => load!(u8, |v| CoordinateValue::Unsigned(v.into())),
            "uint16" => load!(u16, |v| CoordinateValue::Unsigned(v.into())),
            "uint32" => load!(u32, |v| CoordinateValue::Unsigned(v.into())),
            "uint64" => load!(u64, CoordinateValue::Unsigned),
            "int8" => load!(i8, |v| CoordinateValue::Signed(v.into())),
            "int16" => load!(i16, |v| CoordinateValue::Signed(v.into())),
            "int32" => load!(i32, |v| CoordinateValue::Signed(v.into())),
            "int64" => load!(i64, CoordinateValue::Signed),
            "float32" => load!(f32, |v| CoordinateValue::Float(v.into())),
            "float64" => load!(f64, CoordinateValue::Float),
            other => {
                return Err(PbzError::UnsupportedEncoding(format!(
                    "{}: unsupported coordinate type {other}",
                    info.name
                )));
            }
        })
    }
}
/// Storage fill alone never marks a value as missing.
pub(super) fn missing(info: &VariableInfo, value: &CoordinateValue) -> bool {
    let fill = &info.attrs()["_FillValue"];
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
    match value {
        CoordinateValue::String(v) => fill.as_str() == Some(v),
        CoordinateValue::Signed(v) => integer(fill) == Some(i128::from(*v)),
        CoordinateValue::Unsigned(v) => integer(fill) == Some(i128::from(*v)),
        CoordinateValue::Float(v) => v.is_nan() || float_fill(info).ok().flatten() == Some(*v),
    }
}
use std::ops::Range;
use zarrs::array::ElementOwned;

impl Dataset {
    /// Read physical values in a rectangular selection, preserving all axes.
    ///
    /// Missing sentinels, scaling and Xarray logical encodings are not applied.
    /// Supply one half-open range per stored dimension, or [] for a scalar.
    /// Handles/reference metadata are cached; reopen after external mutation.
    pub fn read_raw<T: ElementOwned>(
        &self,
        name: &str,
        ranges: &[Range<u64>],
    ) -> Result<ArrayD<T>> {
        self.read(name, ranges, true)
    }

    /// `read_raw` without the scope check, for a caller that ran
    /// [`Dataset::check_scope`] once and then reads many ranges.
    pub(crate) fn read_checked<T: ElementOwned>(
        &self,
        name: &str,
        ranges: &[Range<u64>],
    ) -> Result<ArrayD<T>> {
        self.read(name, ranges, false)
    }

    /// The validation `read_raw` runs before every read of `name`.
    pub(crate) fn check_scope(&self, name: &str) -> Result<()> {
        let report = self.validate_scope(Some(name))?;
        if report.status == ValidationStatus::Valid {
            return Ok(());
        }
        let message = report
            .issues
            .iter()
            .map(|i| format!("{}: {}", i.rule, i.message))
            .collect::<Vec<_>>()
            .join("; ");
        Err(if report.status == ValidationStatus::Incomplete {
            PbzError::UnsupportedEncoding(message)
        } else {
            PbzError::Metadata(message)
        })
    }

    fn read<T: ElementOwned>(
        &self,
        name: &str,
        ranges: &[Range<u64>],
        check: bool,
    ) -> Result<ArrayD<T>> {
        let info = self
            .variables
            .get(name)
            .ok_or_else(|| PbzError::Metadata(format!("unknown array: {name}")))?;
        if ranges.len() != info.shape.len()
            || ranges
                .iter()
                .zip(&info.shape)
                .any(|(r, &n)| r.start > r.end || r.end > n)
        {
            return Err(PbzError::InvalidRegion {
                message: format!("selection outside {name} shape {:?}", info.shape),
            });
        }
        let shape =
            checked_shape::<T>(&ranges.iter().map(|r| r.end - r.start).collect::<Vec<_>>())?;
        let array = self.array(name)?;
        T::validate_data_type(array.data_type()).map_err(|e| PbzError::InvalidDtype {
            dtype: e.to_string(),
        })?;
        if check {
            self.check_scope(name)?;
        }
        if shape.contains(&0) {
            return ArrayD::from_shape_vec(IxDyn(&shape), Vec::<T>::new())
                .map_err(|e| PbzError::Metadata(e.to_string()));
        }
        array
            .retrieve_array_subset::<ArrayD<T>>(&ArraySubset::new_with_ranges(ranges))
            .map_err(|e| PbzError::Store(format!("read {name}: {e}")))
    }
    /// Decode physical Boolean or unmasked Xarray int8/dtype="bool" values.
    /// Missing Boolean encodings require a representation not yet supported here.
    pub fn read_bool(&self, name: &str, ranges: &[Range<u64>]) -> Result<ArrayD<bool>> {
        let info = self
            .variables
            .get(name)
            .ok_or_else(|| PbzError::Metadata(format!("unknown array: {name}")))?;
        if info.attrs().get("_FillValue").is_some_and(|v| !v.is_null()) {
            return Err(PbzError::UnsupportedEncoding(format!(
                "{name}: masked Boolean encoding"
            )));
        }
        if let Some(reason) = super::metadata::encoding_limit(info) {
            return Err(PbzError::UnsupportedEncoding(format!("{name}: {reason}")));
        }
        if info.metadata["data_type"] == "bool" {
            return self.read_raw::<bool>(name, ranges);
        }
        if info.metadata["data_type"] == "int8" && info.attrs()["dtype"] == "bool" {
            return Ok(self.read_raw::<i8>(name, ranges)?.mapv(|v| v != 0));
        }
        Err(PbzError::UnsupportedEncoding(format!(
            "{name}: no supported Boolean encoding"
        )))
    }
}
