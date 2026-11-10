//! Track metadata + I/O.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use zarrs::array::{Array, ArrayShardedExt, ArraySubset};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::Result;
pub use crate::array::TrackConfig;
use crate::error::PbzError;
use crate::genome::{Genome, Region};
use crate::io::{Dtype, Numeric};
use crate::{PBZ_FORMAT_VERSION, PERBASE_CONVENTION_NAME, PERBASE_CONVENTION_UUID};

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
    /// Whether this track carries a published `multiscales` pyramid. Set from
    /// the group attrs at open (and by `scale` after publication); a sealed
    /// track rejects base writes until the pyramid is unpublished.
    pub(crate) sealed: AtomicBool,
}

pub(crate) struct PendingTrack {
    pub(crate) track: Track,
    completion_attrs: Map<String, Value>,
}

impl PendingTrack {
    pub(crate) fn new(track: Track, config: &TrackConfig) -> Result<Self> {
        let mut completion_attrs = config.extra.clone();
        let attrs = PerbaseTrackAttrs::new(&track.genome, config);
        let attr_val = serde_json::to_value(attrs)
            .map_err(|e| PbzError::Metadata(format!("track {:?}: {e}", track.name)))?;
        let attr_obj = attr_val.as_object().ok_or_else(|| {
            PbzError::Metadata(format!(
                "track {:?}: invalid completion metadata",
                track.name
            ))
        })?;
        for (key, value) in attr_obj {
            completion_attrs.insert(key.clone(), value.clone());
        }
        Ok(Self {
            track,
            completion_attrs,
        })
    }

    pub(crate) fn finalize(self) -> Result<Track> {
        let Self {
            track,
            completion_attrs,
        } = self;
        let mut group =
            zarrs::group::Group::open(Arc::clone(&track.storage), &format!("/{}", track.name))
                .map_err(|e| PbzError::Store(e.to_string()))?;
        for (key, value) in completion_attrs {
            group.attributes_mut().insert(key, value);
        }
        group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Ok(track)
    }
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

    /// Rank of this track: 1 for scalar, 2 for cohort (has `column_dim`).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The column dimension name if this is a cohort track.
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

    /// Decode unit of the `values` array as (positions, columns): the
    /// subchunk when sharded, else the chunk. Columns is 1 for rank-1 tracks.
    pub(crate) fn inner_chunk_shape(&self) -> Result<(u64, u64)> {
        let rank2 = self.rank == 2;
        let values = self.values_array()?;
        if values.is_sharded() {
            let sub = values.effective_subchunk_shape().ok_or_else(|| {
                PbzError::Metadata(format!(
                    "track {:?}: sharded values array with indeterminate subchunk shape",
                    self.name
                ))
            })?;
            let sub = sub.as_slice();
            Ok((sub[0].get(), if rank2 { sub[1].get() } else { 1 }))
        } else {
            let unit = self.write_unit_shape()?;
            Ok((unit[0] as u64, if rank2 { unit[1] as u64 } else { 1 }))
        }
    }

    /// Number of columns: 1 for scalar (rank-1) tracks; for cohort (rank-2)
    /// tracks, reads shape[1] from the `values` array.
    pub fn columns_count(&self) -> Result<usize> {
        if self.rank == 1 {
            return Ok(1);
        }
        Ok(self.values_array()?.shape()[1] as usize)
    }

    /// Read the persisted labels for a rank-2 track's non-position axis.
    pub fn column_labels(&self) -> Result<Vec<String>> {
        if self.rank != 2 {
            return Err(PbzError::Metadata(format!(
                "track {:?} is rank {}, not rank 2",
                self.name, self.rank
            )));
        }
        let dimension = self.column_dim.as_deref().ok_or_else(|| {
            PbzError::Metadata(format!("track {:?} has no column axis", self.name))
        })?;
        let values = self.values_array()?;
        let values_shape = values.shape();
        if values_shape.len() != 2 {
            return Err(PbzError::Metadata(format!(
                "track {:?} values must be rank 2, found rank {}",
                self.name,
                values_shape.len()
            )));
        }
        let width = values_shape[1];
        let path = format!("/{}/{dimension}", self.name);
        let labels = Array::open(Arc::clone(&self.storage), &path)
            .map_err(|error| PbzError::Store(format!("open {path}: {error}")))?;
        let label_shape = labels.shape();
        if label_shape.len() != 1 {
            return Err(PbzError::Metadata(format!(
                "track {:?} column coordinate {dimension:?} must be rank 1, found rank {}",
                self.name,
                label_shape.len()
            )));
        }
        let expected_dimension_names = Some(vec![Some(dimension.to_owned())]);
        if labels.dimension_names() != &expected_dimension_names {
            return Err(PbzError::Metadata(format!(
                "track {:?} column coordinate {dimension:?} has dimension_names {:?}, expected {:?} to match the track column dimension",
                self.name,
                labels.dimension_names(),
                expected_dimension_names
            )));
        }
        let label_count = label_shape[0];
        if label_count != width {
            return Err(PbzError::Metadata(format!(
                "track {:?} column coordinate {dimension:?} has {label_count} labels for {width} columns",
                self.name
            )));
        }
        if label_count == 0 {
            return Ok(Vec::new());
        }
        let subset = ArraySubset::new_with_shape(label_shape.to_vec());
        labels
            .retrieve_array_subset::<ArrayD<String>>(&subset)
            .map(|array| array.into_raw_vec_and_offset().0)
            .map_err(|error| PbzError::Store(format!("read {path}: {error}")))
    }

    /// Shape of one independently writable outer unit: a regular chunk, or a
    /// shard when sharding is configured.
    pub(crate) fn write_unit_shape(&self) -> Result<Vec<usize>> {
        let array = self.values_array()?;
        array
            .chunk_shape_usize(&vec![0; self.rank])
            .map_err(|e| PbzError::Store(format!("write-unit shape for {}: {e}", self.name)))
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

    /// Mark this handle as carrying a published `multiscales` pyramid.
    pub(crate) fn seal(&self) {
        self.sealed.store(true, Ordering::Release);
    }

    /// The seal: base writes on a track with a published pyramid are an error
    /// (the levels would silently go stale). Callers must unpublish first.
    fn ensure_unsealed(&self) -> Result<()> {
        if self.sealed.load(Ordering::Acquire) {
            return Err(PbzError::Metadata(format!(
                "track {:?} has a published multiscales pyramid; base writes are sealed",
                self.name
            )));
        }
        Ok(())
    }

    /// Flat base offset of a region's contig on the `values` position axis.
    ///
    /// Validates the region against its contig: the range must be non-empty
    /// and `region.end` must not exceed the contig length. Without this check
    /// a too-long region would silently read or write the next contig's data.
    fn base_of(&self, region: &Region) -> Result<u64> {
        let contig = self
            .genome
            .get(region.contig)
            .ok_or_else(|| PbzError::InvalidRegion {
                message: format!("unknown contig id {:?}", region.contig),
            })?;
        if region.start >= region.end {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "empty range {}..{} on contig {:?} (length {})",
                    region.start, region.end, contig.name, contig.length
                ),
            });
        }
        if region.end > contig.length {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "range {}..{} extends past contig {:?} (length {})",
                    region.start, region.end, contig.name, contig.length
                ),
            });
        }
        let offsets = self.genome.offsets();
        Ok(offsets[region.contig.as_usize()] as u64)
    }

    /// Read an arbitrary region. Returns an `ArrayD<T>` whose rank matches the
    /// track: shape `[len]` for scalar tracks, `[len, n_cols]` for cohort tracks.
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

    /// Read one rectangular tile of a rank-2 track using flat position and
    /// column ranges. The scale engine uses this to bound slab memory on wide
    /// cohort tracks (whole bins x a column subrange per read).
    pub(crate) fn read_flat_columns<T: Numeric>(
        &self,
        position: std::ops::Range<u64>,
        columns: std::ops::Range<u64>,
    ) -> Result<ArrayD<T>> {
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
        if self.rank != 2 {
            return Err(PbzError::Metadata(format!(
                "column tile read for track {:?} requires a rank-2 track",
                self.name
            )));
        }
        let array = self.values_array()?;
        let shape = array.shape();
        if position.start > position.end
            || columns.start > columns.end
            || position.end > shape[0]
            || columns.end > shape[1]
        {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "column tile {:?} × {:?} is outside track {:?}",
                    position, columns, self.name
                ),
            });
        }
        let subset = ArraySubset::new_with_ranges(&[position, columns]);
        array
            .retrieve_array_subset::<ArrayD<T>>(&subset)
            .map_err(|e| PbzError::Store(format!("read {}: {e}", self.name)))
    }

    /// Read a half-open flat position range, which may span contig
    /// boundaries. Full column width for rank-2 tracks.
    pub(crate) fn read_flat<T: Numeric>(
        &self,
        position: std::ops::Range<u64>,
    ) -> Result<ArrayD<T>> {
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
        let array = self.values_array()?;
        let shape = array.shape();
        if position.start >= position.end || position.end > shape[0] {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "flat range {}..{} is outside track {:?}",
                    position.start, position.end, self.name
                ),
            });
        }
        #[allow(clippy::single_range_in_vec_init)]
        let subset = if self.rank == 1 {
            ArraySubset::new_with_ranges(&[position])
        } else {
            ArraySubset::new_with_ranges(&[position, 0..shape[1]])
        };
        array
            .retrieve_array_subset::<ArrayD<T>>(&subset)
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
        self.ensure_unsealed()?;
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
        if self.rank == 2 {
            return self.write_flat_columns(start..end, 0..self.columns_count()? as u64, data);
        }
        let arr = self.values_array()?;
        if start > end || end > arr.shape()[0] {
            return Err(PbzError::InvalidRegion {
                message: format!("flat range {start}..{end} is outside track {:?}", self.name),
            });
        }
        if data.shape()
            != [usize::try_from(end - start)
                .map_err(|_| PbzError::Metadata("flat write length exceeds usize".into()))?]
        {
            return Err(PbzError::Metadata(format!(
                "shape mismatch for track {:?}: expected [{}], got {:?}",
                self.name,
                end - start,
                data.shape(),
            )));
        }
        #[allow(clippy::single_range_in_vec_init)]
        let subset = ArraySubset::new_with_ranges(&[start..end]);
        arr.store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }

    /// Write one rectangular tile of a rank-2 track using flat position and
    /// column ranges. Import workers use full write units so they never race.
    pub(crate) fn write_flat_columns<T: Numeric>(
        &self,
        position: std::ops::Range<u64>,
        columns: std::ops::Range<u64>,
        data: ArrayD<T>,
    ) -> Result<()> {
        self.ensure_unsealed()?;
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
        if self.rank != 2 || data.ndim() != 2 {
            return Err(PbzError::Metadata(format!(
                "column tile for track {:?} requires rank-2 data",
                self.name
            )));
        }
        let array = self.values_array()?;
        let shape = array.shape();
        if position.start > position.end
            || columns.start > columns.end
            || position.end > shape[0]
            || columns.end > shape[1]
        {
            return Err(PbzError::InvalidRegion {
                message: format!(
                    "column tile {:?} × {:?} is outside track {:?}",
                    position, columns, self.name
                ),
            });
        }
        let rows = usize::try_from(position.end - position.start)
            .map_err(|_| PbzError::Metadata("tile row count exceeds usize".into()))?;
        let width = usize::try_from(columns.end - columns.start)
            .map_err(|_| PbzError::Metadata("tile column count exceeds usize".into()))?;
        if data.shape() != [rows, width] {
            return Err(PbzError::Metadata(format!(
                "shape mismatch for track {:?}: expected [{rows}, {width}], got {:?}",
                self.name,
                data.shape(),
            )));
        }
        let subset = ArraySubset::new_with_ranges(&[position, columns]);
        array
            .store_array_subset(&subset, data)
            .map_err(|e| PbzError::Store(format!("write {}: {e}", self.name)))
    }
}
