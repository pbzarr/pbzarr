//! Array settings shared by the store writers: track config, node names,
//! dtypes, codecs and fill values.

use std::sync::Arc;

use serde_json::{Map, Value};
use zarrs::array::codec::api::BytesToBytesCodecTraits;
use zarrs::array::codec::{
    BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode, BytesCodec,
};
use zarrs::array::{ChunkGrid, FillValue, data_type};
use zarrs::node::NodeName;

use crate::Result;
use crate::codec_spec::{ExplicitArraySpec, resolve_codecs};
use crate::error::PbzError;
use crate::io::{Dtype, Null};

/// Validate a non-root PBZ/Zarr hierarchy node name using zarrs' canonical
/// Zarr v3 rules.
pub fn validate_node_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(PbzError::Metadata(
            "invalid PBZ/Zarr node name \"\": direct-child names cannot be empty".into(),
        ));
    }
    NodeName::new(name.to_owned()).map(|_| ()).map_err(|error| {
        PbzError::Metadata(format!("invalid PBZ/Zarr node name {name:?}: {error}"))
    })
}

/// The value a track holds at positions that no source covers.
#[derive(Debug, Clone, PartialEq)]
pub enum Fill {
    /// A JSON scalar, coerced to the track dtype.
    Value(Value),
    /// The null of the track dtype (see [`Dtype::null`]). The track also
    /// declares the null as its `_FillValue` attribute.
    Na,
}

/// User-supplied configuration for a new track.
#[derive(Debug, Clone)]
pub struct TrackConfig {
    pub dtype: Dtype,
    pub chunk_size: usize,
    pub column_dim: Option<String>,
    pub columns: Option<Vec<String>>,
    pub column_chunk_size: Option<usize>,
    pub shard_size: Option<usize>,
    pub shard_column_size: Option<usize>,
    /// Explicit Zarr codec/chunk-grid metadata for the track array.
    /// Replaces the default Blosc pipeline; the chunk/shard size fields are
    /// then ignored except as the fallback outer grid when the spec carries
    /// no `chunk_grid`.
    pub codecs: Option<ExplicitArraySpec>,
    /// `None` gives the dtype default: 0 for integers, NaN for floats, false
    /// for bool, with no `_FillValue`.
    pub fill: Option<Fill>,
    pub description: Option<String>,
    pub source: Option<String>,
    pub extra: Map<String, Value>,
}

impl TrackConfig {
    /// Create a 1D track configuration with default chunk size (1M positions).
    /// Call `.columns(...)` to make it 2D.
    pub fn new(dtype: Dtype) -> Self {
        Self {
            dtype,
            chunk_size: 1_000_000,
            column_dim: None,
            columns: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            codecs: None,
            fill: None,
            description: None,
            source: None,
            extra: Map::new(),
        }
    }

    /// Add a column axis to this track. The track becomes 2D with shape
    /// `(position, column)`. `column_dim` defaults to `"column"`; override
    /// with `.column_dim(...)`. `column_chunk_size` defaults to 16.
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        if self.column_dim.is_none() {
            self.column_dim = Some("column".into());
        }
        if self.column_chunk_size.is_none() {
            self.column_chunk_size = Some(16);
        }
        self
    }

    /// Override the column-axis dimension name (e.g., `"sample"`).
    /// No effect for 1D tracks.
    pub fn column_dim(mut self, name: impl Into<String>) -> Self {
        self.column_dim = Some(name.into());
        self
    }

    pub fn chunk_size(mut self, n: usize) -> Self {
        self.chunk_size = n;
        self
    }

    pub fn column_chunk_size(mut self, n: usize) -> Self {
        self.column_chunk_size = Some(n);
        self
    }

    pub fn shard_size(mut self, n: usize) -> Self {
        self.shard_size = Some(n);
        self
    }

    pub fn shard_column_size(mut self, n: usize) -> Self {
        self.shard_column_size = Some(n);
        self
    }

    pub fn codecs(mut self, spec: ExplicitArraySpec) -> Self {
        self.codecs = Some(spec);
        self
    }

    pub fn fill(mut self, fill: Fill) -> Self {
        self.fill = Some(fill);
        self
    }

    pub fn fill_value(self, v: serde_json::Value) -> Self {
        self.fill(Fill::Value(v))
    }

    pub fn description(mut self, s: impl Into<String>) -> Self {
        self.description = Some(s.into());
        self
    }

    pub fn source(mut self, s: impl Into<String>) -> Self {
        self.source = Some(s.into());
        self
    }
}

/// Default compression: Blosc(zstd, level 5, byte shuffle), per the pbzarr spec.
///
/// Built per-array so `typesize` matches the element size. Returns the
/// vec-form that `ArrayBuilder::bytes_to_bytes_codecs` consumes.
pub(crate) fn default_data_codecs(dtype: Dtype) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>> {
    let typesize = dtype.size();
    let clevel = BloscCompressionLevel::try_from(5u8)
        .map_err(|e| PbzError::Store(format!("invalid blosc clevel: {e}")))?;
    let codec = BloscCodec::new(
        BloscCompressor::Zstd,
        clevel,
        None,
        BloscShuffleMode::Shuffle,
        Some(typesize),
    )
    .map_err(|e| PbzError::Store(format!("blosc codec init: {e}")))?;
    Ok(vec![Arc::new(codec)])
}

/// Apply a user-supplied explicit spec to `builder`: the resolved codec
/// chain (default little-endian `bytes` inserted when the spec names no
/// array-to-bytes codec), plus the chunk grid override when present.
pub(crate) fn apply_explicit_spec(
    builder: &mut zarrs::array::ArrayBuilder,
    track: &str,
    spec: &ExplicitArraySpec,
    array_shape: &[u64],
) -> Result<()> {
    let resolved = resolve_codecs(track, &spec.codecs)?;
    builder.array_to_array_codecs(resolved.array_to_array);
    builder.array_to_bytes_codec(
        resolved
            .array_to_bytes
            .unwrap_or_else(|| Arc::new(BytesCodec::little())),
    );
    builder.bytes_to_bytes_codecs(resolved.bytes_to_bytes);
    if let Some(grid_metadata) = &spec.chunk_grid {
        let grid = ChunkGrid::from_metadata(grid_metadata, array_shape).map_err(|e| {
            PbzError::Metadata(format!(
                "track {track:?}: chunk_grid for a rank-{} track array: {e}",
                array_shape.len()
            ))
        })?;
        builder.chunk_grid(grid);
    }
    Ok(())
}

/// Map `Dtype` → a zarrs `DataType` value.
pub(crate) fn dtype_to_zarrs(d: Dtype) -> zarrs::array::DataType {
    match d {
        Dtype::U8 => data_type::uint8(),
        Dtype::U16 => data_type::uint16(),
        Dtype::U32 => data_type::uint32(),
        Dtype::U64 => data_type::uint64(),
        Dtype::I8 => data_type::int8(),
        Dtype::I16 => data_type::int16(),
        Dtype::I32 => data_type::int32(),
        Dtype::I64 => data_type::int64(),
        Dtype::F32 => data_type::float32(),
        Dtype::F64 => data_type::float64(),
        Dtype::Bool => data_type::bool(),
    }
}

/// The Zarr fill value of a track, plus the `_FillValue` attribute when the
/// fill is the null.
pub(crate) fn resolve_fill(d: Dtype, fill: Option<&Fill>) -> Result<(FillValue, Option<Value>)> {
    match fill {
        None => Ok((default_fill_value(d), None)),
        Some(Fill::Value(v)) => Ok((resolve_fill_value(d, Some(v))?, None)),
        Some(Fill::Na) => {
            let null = d.null().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "a {d} track has no null; give a fill of true or false"
                ))
            })?;
            let fill = match null {
                Null::Nan => default_fill_value(d),
                Null::Int(_) => resolve_fill_value(d, Some(&null.attribute()))?,
            };
            Ok((fill, Some(null.attribute())))
        }
    }
}

/// Coerce a JSON scalar fill to the track dtype, or give the dtype default.
pub(crate) fn resolve_fill_value(d: Dtype, custom: Option<&Value>) -> Result<FillValue> {
    let Some(v) = custom else {
        return Ok(default_fill_value(d));
    };
    let bad = || PbzError::Metadata(format!("fill_value {v} out of range for {d} track"));
    let fv = match d {
        Dtype::U8 => {
            FillValue::from(u8::try_from(v.as_u64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::U16 => {
            FillValue::from(u16::try_from(v.as_u64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::U32 => {
            FillValue::from(u32::try_from(v.as_u64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::U64 => FillValue::from(v.as_u64().ok_or_else(&bad)?),
        Dtype::I8 => {
            FillValue::from(i8::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::I16 => {
            FillValue::from(i16::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::I32 => {
            FillValue::from(i32::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::I64 => FillValue::from(v.as_i64().ok_or_else(&bad)?),
        Dtype::F32 => FillValue::from(v.as_f64().ok_or_else(&bad)? as f32),
        Dtype::F64 => FillValue::from(v.as_f64().ok_or_else(&bad)?),
        Dtype::Bool => FillValue::from(v.as_bool().ok_or_else(&bad)?),
    };
    Ok(fv)
}

/// Default fill value for a dtype: 0 for ints, NaN for floats, false for bool.
fn default_fill_value(d: Dtype) -> FillValue {
    match d {
        Dtype::U8 => FillValue::from(0u8),
        Dtype::U16 => FillValue::from(0u16),
        Dtype::U32 => FillValue::from(0u32),
        Dtype::U64 => FillValue::from(0u64),
        Dtype::I8 => FillValue::from(0i8),
        Dtype::I16 => FillValue::from(0i16),
        Dtype::I32 => FillValue::from(0i32),
        Dtype::I64 => FillValue::from(0i64),
        Dtype::F32 => FillValue::from(f32::NAN),
        Dtype::F64 => FillValue::from(f64::NAN),
        Dtype::Bool => FillValue::from(false),
    }
}

#[cfg(test)]
mod tests {
    use super::dtype_to_zarrs;
    use crate::io::Dtype;

    // The reopen path (`Dtype::from_zarrs`) is only exercised through I32
    // tracks elsewhere in the test suite; this closes the gap for every
    // dtype the store can write.
    #[test]
    fn dtype_round_trips_through_zarrs_data_type() {
        let all = [
            Dtype::U8,
            Dtype::U16,
            Dtype::U32,
            Dtype::U64,
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
            Dtype::I64,
            Dtype::F32,
            Dtype::F64,
            Dtype::Bool,
        ];
        for d in all {
            let zdt = dtype_to_zarrs(d);
            assert_eq!(Dtype::from_zarrs(&zdt).unwrap(), d);
        }
    }
}
