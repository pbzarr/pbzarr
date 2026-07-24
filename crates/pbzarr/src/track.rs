//! Track metadata + I/O.

use std::sync::{Arc, RwLock};

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::Result;
use crate::error::PbzError;
use crate::genome::{Genome, Region};
use crate::io::{Dtype, Numeric};
use crate::{PBZ_FORMAT_VERSION, PERBASE_CONVENTION_NAME, PERBASE_CONVENTION_UUID};

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

/// One entry of the `zarr_conventions` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConventionRef {
    pub uuid: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_url: Option<String>,
}

/// The interpretation block written on a track group's `zarr.json`. Structural
/// facts (dtype, chunks, shards, dim names, labels) are recovered from the
/// arrays and are deliberately absent here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerbaseTrackAttrs {
    pub zarr_conventions: Vec<ConventionRef>,
    #[serde(rename = "perbase:version")]
    pub version: String,
    #[serde(rename = "perbase:genome_checksum")]
    pub genome_checksum: String,
    #[serde(
        rename = "perbase:genome_name",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub genome_name: Option<String>,
    #[serde(rename = "perbase:ragged_index")]
    pub ragged_index: String,
    #[serde(rename = "perbase:ragged_contigs")]
    pub ragged_contigs: String,
    #[serde(rename = "perbase:coordinates")]
    pub coordinates: String,
    #[serde(
        rename = "perbase:description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,
    #[serde(
        rename = "perbase:source",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub source: Option<String>,
}

impl PerbaseTrackAttrs {
    pub fn new(genome: &Genome, config: &TrackConfig) -> Self {
        Self {
            zarr_conventions: vec![ConventionRef {
                uuid: PERBASE_CONVENTION_UUID.to_owned(),
                name: PERBASE_CONVENTION_NAME.to_owned(),
                spec_url: None,
                schema_url: None,
            }],
            version: PBZ_FORMAT_VERSION.to_owned(),
            genome_checksum: genome.checksum(),
            genome_name: genome.name().map(|s| s.to_owned()),
            ragged_index: "offsets".to_owned(),
            ragged_contigs: "contigs".to_owned(),
            coordinates: "0-based-half-open".to_owned(),
            description: config.description.clone(),
            source: config.source.clone(),
        }
    }

    /// Whether a group's attributes declare the `perbase` convention. Used by
    /// store discovery to recognize track groups and to gate on completion
    /// (the block is written last, so its presence means "fully written").
    pub fn conforms(attrs: &Map<String, Value>) -> bool {
        attrs
            .get("zarr_conventions")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter().any(|c| {
                    c.get("name").and_then(|n| n.as_str()) == Some(PERBASE_CONVENTION_NAME)
                })
            })
    }
}

/// Track handle: addressing + I/O over one flat `values` array.
pub struct Track {
    pub(crate) name: String,
    pub(crate) genome: Arc<Genome>,
    pub(crate) dtype: Dtype,
    pub(crate) rank: usize,
    pub(crate) column_dim: Option<String>,
    pub(crate) storage: ReadableWritableListableStorage,
    /// Cached `values` array; `Array::open` re-reads `zarr.json`, so opening
    /// once keeps per-region reads/writes cheap.
    pub(crate) values: RwLock<Option<Arc<Array<dyn ReadableWritableListableStorageTraits>>>>,
}

impl Track {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The genome owned by this track.
    pub fn genome(&self) -> &Arc<Genome> {
        &self.genome
    }

    /// Runtime dtype tag. Validated at `PbzStore::open` / `create_track` time.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Rank of this track: 1 for scalar, 2 for a 2D column track (has `column_dim`).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The column dimension name if this is a 2D track.
    pub fn column_dim(&self) -> Option<&str> {
        self.column_dim.as_deref()
    }

    /// Total flat length `ΣL` of this track (sum of its genome's contig lengths).
    pub fn total_len(&self) -> u64 {
        self.genome.contigs().iter().map(|c| c.length).sum()
    }

    /// Position chunk size for this track, read from the `values` array's
    /// chunk shape at the first chunk (uniform except at the tail).
    pub fn chunk_size(&self) -> Result<usize> {
        let arr = self.values_array()?;
        let indices = vec![0u64; self.rank];
        let shape = arr
            .chunk_shape_usize(&indices)
            .map_err(|e| PbzError::Store(format!("chunk shape for {}: {e}", self.name)))?;
        shape.first().copied().ok_or_else(|| {
            PbzError::Metadata(format!("track {:?}: rank-0 values array", self.name))
        })
    }

    /// Number of columns: 1 for scalar (rank-1) tracks; for 2D (rank-2)
    /// tracks, reads shape[1] from the `values` array.
    pub fn columns_count(&self) -> Result<usize> {
        if self.rank == 1 {
            return Ok(1);
        }
        Ok(self.values_array()?.shape()[1] as usize)
    }

    /// The cached `values` array, opened lazily on first use.
    pub(crate) fn values_array(
        &self,
    ) -> Result<Arc<Array<dyn ReadableWritableListableStorageTraits>>> {
        if let Some(arr) = self.values.read().expect("values lock poisoned").as_ref() {
            return Ok(Arc::clone(arr));
        }
        let mut w = self.values.write().expect("values lock poisoned");
        if let Some(arr) = w.as_ref() {
            return Ok(Arc::clone(arr));
        }
        let path = format!("/{}/values", self.name);
        let arr = Array::open(Arc::clone(&self.storage), &path)
            .map_err(|e| PbzError::Store(format!("open {path}: {e}")))?;
        let arr = Arc::new(arr);
        *w = Some(Arc::clone(&arr));
        Ok(arr)
    }

    /// Flat base offset of a region's contig on the `values` position axis.
    fn base_of(&self, region: &Region) -> Result<u64> {
        let idx = region.contig.as_usize();
        let offsets = self.genome.offsets();
        offsets
            .get(idx)
            .map(|b| *b as u64)
            .ok_or_else(|| PbzError::InvalidRegion {
                message: format!("unknown contig id {:?}", region.contig),
            })
    }

    /// Read an arbitrary region. Returns an `ArrayD<T>` whose rank matches the
    /// track: shape `[len]` for scalar tracks, `[len, n_cols]` for 2D tracks.
    ///
    /// Returns `Err` if `T::DTYPE` doesn't match the track's dtype.
    pub fn read_region<T: Numeric>(&self, region: &Region) -> Result<ArrayD<T>> {
        if T::DTYPE != self.dtype {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but caller requested {}",
                    self.name,
                    self.dtype,
                    T::DTYPE
                ),
            });
        }
        let base = self.base_of(region)?;
        let arr = self.values_array()?;
        let (start, end) = (base + region.start, base + region.end);
        #[allow(clippy::single_range_in_vec_init)]
        let subset = if self.rank == 1 {
            ArraySubset::new_with_ranges(&[start..end])
        } else {
            let n_cols = arr.shape()[1];
            ArraySubset::new_with_ranges(&[start..end, 0..n_cols])
        };
        arr.retrieve_array_subset::<ArrayD<T>>(&subset)
            .map_err(|e| PbzError::Store(format!("read {}: {e}", self.name)))
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
        let base = self.base_of(region)?;
        self.write_flat(base + region.start, base + region.end, data)
    }

    /// Write `data` into the half-open flat position range `[start, end)`,
    /// which may span contig boundaries. The import pipeline uses this to write
    /// one physical chunk at a time regardless of where contigs begin.
    pub(crate) fn write_flat<T: Numeric>(
        &self,
        start: u64,
        end: u64,
        data: ArrayD<T>,
    ) -> Result<()> {
        if T::DTYPE != self.dtype {
            return Err(PbzError::InvalidDtype {
                dtype: format!(
                    "track {:?} is {} but caller wrote {}",
                    self.name,
                    self.dtype,
                    T::DTYPE
                ),
            });
        }
        if data.ndim() != self.rank {
            return Err(PbzError::Metadata(format!(
                "rank mismatch for track {:?}: expected {} got {}",
                self.name,
                self.rank,
                data.ndim(),
            )));
        }
        let arr = self.values_array()?;
        #[allow(clippy::single_range_in_vec_init)]
        let subset = if self.rank == 1 {
            ArraySubset::new_with_ranges(&[start..end])
        } else {
            ArraySubset::new_with_ranges(&[start..end, 0..(data.shape()[1] as u64)])
        };
        arr.store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Contig;

    #[test]
    fn perbase_attrs_roundtrip_and_conform() {
        let g = Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 100,
        }])
        .unwrap()
        .with_name("hg38");
        let cfg = TrackConfig::new(Dtype::I32);
        let attrs = PerbaseTrackAttrs::new(&g, &cfg);

        let val = serde_json::to_value(&attrs).unwrap();
        let obj = val.as_object().unwrap();
        assert!(PerbaseTrackAttrs::conforms(obj));
        assert_eq!(obj["perbase:version"], "0.4");
        assert_eq!(obj["perbase:genome_name"], "hg38");
        assert_eq!(obj["perbase:ragged_index"], "offsets");
        assert!(obj["zarr_conventions"].is_array());

        // A plain group without the marker does not conform.
        let plain = serde_json::json!({"foo": 1});
        assert!(!PerbaseTrackAttrs::conforms(plain.as_object().unwrap()));

        let back: PerbaseTrackAttrs = serde_json::from_value(val).unwrap();
        assert_eq!(back.version, "0.4");
        assert_eq!(back.genome_checksum, g.checksum());
    }
}
