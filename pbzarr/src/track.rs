//! Track metadata + I/O. See `docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md`.

use std::sync::{Arc, RwLock};

use hashbrown::HashMap;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zarrs::array::{Array, ArraySubset};
use zarrs::filesystem::FilesystemStore;

use crate::Result;
use crate::error::PbzError;
use crate::genome::{Genome, Region};
use crate::io::{Dtype, Numeric};

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
            fill_value: None,
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

    pub fn fill_value(mut self, v: serde_json::Value) -> Self {
        self.fill_value = Some(v);
        self
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
    /// Parsed dtype tag. Validated at construction so `dtype()` is panic-free.
    pub(crate) dtype: Dtype,
    /// Concrete filesystem store; shared via Arc with the owning PbzStore.
    pub(crate) fs: Arc<FilesystemStore>,
    /// Genome shared with the owning PbzStore; needed to map ContigId → name.
    pub(crate) genome: Arc<Genome>,
    /// Per-contig `zarrs::Array` cache. `Array::open` re-reads `zarr.json`
    /// from disk, so without this every read/write would do that — turning a
    /// 250-chunk-per-contig ingest into 250 metadata reloads per track.
    pub(crate) arrays: RwLock<HashMap<String, Arc<Array<FilesystemStore>>>>,
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

    /// Runtime dtype tag. Validated at `PbzStore::open` / `create_track` time.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Rank of this track: 1 for scalar, 2 for cohort (has `column_dim`).
    pub fn rank(&self) -> usize {
        if self.metadata.column_dim.is_some() {
            2
        } else {
            1
        }
    }

    /// The column dimension name if this is a cohort track.
    pub fn column_dim(&self) -> Option<&str> {
        self.metadata.column_dim.as_deref()
    }

    /// Position chunk size for this track.
    pub fn chunk_size(&self) -> usize {
        self.metadata.chunk_size
    }

    /// Number of columns: 1 for scalar (rank-1) tracks; for cohort (rank-2)
    /// tracks, reads shape[1] from the zarr array on the first contig.
    ///
    /// Returns `Err` if the store has no contigs (degenerate case) or the
    /// underlying zarrs `Array::open` fails.
    pub fn columns_count(&self) -> Result<usize> {
        if self.rank() == 1 {
            return Ok(1);
        }
        let first = self.genome.contigs().first().ok_or_else(|| {
            PbzError::Metadata(format!(
                "track {:?}: cannot determine column count, store has no contigs",
                self.name
            ))
        })?;
        let arr = self.array_for(&first.name)?;
        Ok(arr.shape()[1] as usize)
    }

    /// Return the cached `Array` handle for `contig_name`, opening it lazily
    /// on first use. Concurrent reads share the cache without serializing.
    fn array_for(&self, contig_name: &str) -> Result<Arc<Array<FilesystemStore>>> {
        if let Some(arr) = self
            .arrays
            .read()
            .expect("track array cache lock poisoned")
            .get(contig_name)
        {
            return Ok(Arc::clone(arr));
        }
        let mut w = self
            .arrays
            .write()
            .expect("track array cache lock poisoned");
        // Re-check under the write lock in case another thread won the race.
        if let Some(arr) = w.get(contig_name) {
            return Ok(Arc::clone(arr));
        }
        let path = format!("/{}/{}", contig_name, self.name);
        let arr = Array::open(Arc::clone(&self.fs), &path)
            .map_err(|e| PbzError::Store(format!("open {path}: {e}")))?;
        let arr = Arc::new(arr);
        w.insert(contig_name.to_owned(), Arc::clone(&arr));
        Ok(arr)
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
        let arr = self.array_for(&contig.name)?;

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
    /// Takes ownership of `data` because zarrs' `store_array_subset` consumes
    /// the buffer; passing by value lets the pipeline writer avoid a per-chunk
    /// clone. Partial-chunk writes trigger a read-modify-write internally
    /// inside zarrs.
    ///
    /// Returns `Err` if `T::DTYPE` doesn't match the track dtype, or if the
    /// data rank doesn't match the track rank.
    pub fn write_region<T: Numeric>(&self, region: &Region, data: ArrayD<T>) -> Result<()> {
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
        let arr = self.array_for(&contig.name)?;

        #[allow(clippy::single_range_in_vec_init)]
        let subset = if expected_rank == 1 {
            // Single-element range array is intentional: new_with_ranges takes &[Range<u64>].
            let ranges = [region.start..region.end];
            ArraySubset::new_with_ranges(&ranges)
        } else {
            ArraySubset::new_with_ranges(&[region.start..region.end, 0..(data.shape()[1] as u64)])
        };
        arr.store_array_subset(&subset, data)
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
