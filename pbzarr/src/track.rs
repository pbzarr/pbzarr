//! Track metadata + I/O. See `docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md`.

use crate::io::Dtype;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
    pub fill_value: Option<Value>,
    pub description: Option<String>,
    pub source: Option<String>,
    pub extra: Map<String, Value>,
}

impl TrackConfig {
    /// 1D scalar track with sensible defaults (chunk_size = 1M).
    pub fn scalar(dtype: Dtype) -> Self {
        Self {
            dtype,
            chunk_size: 1_000_000,
            column_dim: None,
            columns: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            fill_value: None,
            description: None,
            source: None,
            extra: Map::new(),
        }
    }

    /// 2D cohort track with sensible defaults (chunk_size = 1M, column_chunk_size = 16,
    /// column_dim = "sample").
    pub fn cohort(dtype: Dtype, columns: Vec<String>) -> Self {
        Self {
            dtype,
            chunk_size: 1_000_000,
            column_dim: Some("sample".into()),
            columns: Some(columns),
            column_chunk_size: Some(16),
            shard_size: None,
            shard_column_size: None,
            fill_value: None,
            description: None,
            source: None,
            extra: Map::new(),
        }
    }

    /// True if `columns` is set (i.e., 2D cohort track).
    pub fn is_cohort(&self) -> bool {
        self.columns.is_some()
    }
}

/// On-disk track metadata as it appears in `root.perbase_zarr.tracks[name]`.
/// Round-trippable via serde.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    pub dtype: String,
    pub chunk_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_dim: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_chunk_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard_column_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_value: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Track handle. Real I/O methods are added in Task 8.
pub struct Track {
    pub(crate) name: String,
    pub(crate) metadata: TrackMetadata,
}

impl Track {
    pub fn name(&self) -> &str { &self.name }
    pub fn metadata(&self) -> &TrackMetadata { &self.metadata }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrip_scalar_and_cohort() {
        let scalar = TrackMetadata {
            dtype: "bool".into(),
            chunk_size: 1_000_000,
            column_dim: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            fill_value: None,
            description: None,
            source: None,
            extra: Map::new(),
        };
        let json = serde_json::to_string(&scalar).unwrap();
        let back: TrackMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dtype, "bool");
        assert!(back.column_dim.is_none());
        // optional fields elided from JSON
        assert!(!json.contains("column_dim"));
        assert!(!json.contains("fill_value"));

        let cohort = TrackMetadata {
            dtype: "uint16".into(),
            chunk_size: 1_000_000,
            column_dim: Some("sample".into()),
            column_chunk_size: Some(16),
            shard_size: None,
            shard_column_size: None,
            fill_value: None,
            description: None,
            source: None,
            extra: Map::new(),
        };
        let json = serde_json::to_string(&cohort).unwrap();
        let back: TrackMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.column_dim.as_deref(), Some("sample"));
        assert_eq!(back.column_chunk_size, Some(16));
    }
}
