//! Track metadata + I/O. See `docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md`.

use std::sync::Arc;

use ndarray::{ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zarrs::array::ArraySubset;
use zarrs::filesystem::FilesystemStore;

use crate::error::PbzError;
use crate::genome::{Genome, Region};
use crate::io::{Dtype, Numeric};
use crate::Result;

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

/// Track handle with I/O methods for reading and writing regions.
pub struct Track {
    pub(crate) name: String,
    pub(crate) metadata: TrackMetadata,
    /// Concrete filesystem store; shared via Arc with the owning PbzStore.
    pub(crate) fs: Arc<FilesystemStore>,
    /// Genome shared with the owning PbzStore; needed to map ContigId → name.
    pub(crate) genome: Arc<Genome>,
}

impl Track {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The genome shared with the owning store.
    pub fn genome(&self) -> &Arc<Genome> {
        &self.genome
    }

    pub fn metadata(&self) -> &TrackMetadata {
        &self.metadata
    }

    /// Runtime dtype tag parsed from the on-disk metadata string.
    ///
    /// Panics only if the on-disk dtype string was corrupted past what `open()`
    /// validation would catch — treated as a programming invariant violation.
    pub fn dtype(&self) -> Dtype {
        match self.metadata.dtype.as_str() {
            "uint8" => Dtype::U8,
            "uint16" => Dtype::U16,
            "uint32" => Dtype::U32,
            "int8" => Dtype::I8,
            "int16" => Dtype::I16,
            "int32" => Dtype::I32,
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            "bool" => Dtype::Bool,
            other => panic!("track {:?}: unknown dtype {other}", self.name),
        }
    }

    /// Rank of this track: 1 for scalar, 2 for cohort (has `column_dim`).
    pub fn rank(&self) -> usize {
        if self.metadata.column_dim.is_some() { 2 } else { 1 }
    }

    /// The column dimension name if this is a cohort track.
    pub fn column_dim(&self) -> Option<&str> {
        self.metadata.column_dim.as_deref()
    }

    /// Position chunk size for this track.
    pub fn chunk_size(&self) -> usize {
        self.metadata.chunk_size
    }

    /// Read an arbitrary region. Returns an `ArrayD<T>` whose rank matches the
    /// track: shape `[len]` for scalar tracks, `[len, n_cols]` for cohort tracks.
    ///
    /// Returns `Err` if `T::DTYPE` doesn't match the track's dtype.
    pub fn read_region<T: Numeric>(&self, region: &Region) -> Result<ArrayD<T>> {
        if T::DTYPE != self.dtype() {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but caller requested {}",
                    self.name,
                    self.dtype(),
                    T::DTYPE
                ),
            });
        }
        let contig = self
            .genome
            .get(region.contig)
            .ok_or_else(|| PbzError::InvalidRegion {
                message: format!("unknown contig id {:?}", region.contig),
            })?;
        let path = format!("/{}/{}", contig.name, self.name);
        let arr = zarrs::array::Array::open(Arc::clone(&self.fs), &path)
            .map_err(|e| PbzError::Store(format!("open {path}: {e}")))?;

        #[allow(clippy::single_range_in_vec_init)]
        let subset = if self.rank() == 1 {
            // Single-element range array is intentional: new_with_ranges takes &[Range<u64>].
            let ranges = [region.start..region.end];
            ArraySubset::new_with_ranges(&ranges)
        } else {
            let n_cols = arr.shape()[1];
            ArraySubset::new_with_ranges(&[region.start..region.end, 0..n_cols])
        };
        let nd = arr
            .retrieve_array_subset::<ArrayD<T>>(&subset)
            .map_err(|e| PbzError::Store(format!("read {}: {e}", self.name)))?;
        Ok(nd)
    }

    /// Write data into an arbitrary region.
    ///
    /// Partial-chunk writes trigger a read-modify-write internally inside zarrs.
    /// Returns `Err` if `T::DTYPE` doesn't match the track dtype, or if the
    /// data rank doesn't match the track rank.
    pub fn write_region<T: Numeric>(
        &self,
        region: &Region,
        data: ArrayViewD<'_, T>,
    ) -> Result<()> {
        if T::DTYPE != self.dtype() {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but caller wrote {}",
                    self.name,
                    self.dtype(),
                    T::DTYPE
                ),
            });
        }
        let expected_rank = self.rank();
        if data.ndim() != expected_rank {
            return Err(PbzError::Metadata(format!(
                "rank mismatch for track {:?}: expected {} got {}",
                self.name,
                expected_rank,
                data.ndim(),
            )));
        }
        let contig = self
            .genome
            .get(region.contig)
            .ok_or_else(|| PbzError::InvalidRegion {
                message: format!("unknown contig id {:?}", region.contig),
            })?;
        let path = format!("/{}/{}", contig.name, self.name);
        let arr = zarrs::array::Array::open(Arc::clone(&self.fs), &path)
            .map_err(|e| PbzError::Store(format!("open {path}: {e}")))?;

        #[allow(clippy::single_range_in_vec_init)]
        let subset = if expected_rank == 1 {
            // Single-element range array is intentional: new_with_ranges takes &[Range<u64>].
            let ranges = [region.start..region.end];
            ArraySubset::new_with_ranges(&ranges)
        } else {
            ArraySubset::new_with_ranges(&[
                region.start..region.end,
                0..(data.shape()[1] as u64),
            ])
        };
        // store_array_subset requires owned ndarray; to_owned converts the view.
        arr.store_array_subset(&subset, data.to_owned())
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))?;
        Ok(())
    }
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
