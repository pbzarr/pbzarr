use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD};
use zarrs::array::codec::array_to_bytes::packbits::PackBitsCodec;
use zarrs::array::codec::bytes_to_bytes::blosc::{
    BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
};
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, DataType, ElementOwned, FillValue, data_type,
};
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::{PbzError, Result};

/// Attribute key on track groups identifying them as PBZ tracks.
const TRACK_ATTR_KEY: &str = "perbase_zarr_track";

/// Standard fields in the track metadata attribute (everything else goes into `extra`).
const STANDARD_FIELDS: &[&str] = &[
    "layout",
    "dtype",
    "chunk_size",
    "column_chunk_size",
    "column_dim_name",
    "description",
    "source",
];

/// Default chunk size along the position axis (1 million base pairs).
pub const DEFAULT_CHUNK_SIZE: u64 = 1_000_000;

/// Default chunk size along the columns axis.
pub const DEFAULT_COLUMN_CHUNK_SIZE: u64 = 16;

/// Default Xarray dimension name for the columns axis.
pub const DEFAULT_COLUMN_DIM_NAME: &str = "column";

/// Layout identifier for the per_base track layout (the only layout in spec v0.1).
pub const PER_BASE_LAYOUT: &str = "per_base";

/// Configuration for creating a new track.
pub struct TrackConfig {
    pub dtype: String,
    pub columns: Option<Vec<String>>,
    pub chunk_size: u64,
    pub column_chunk_size: u64,
    /// Xarray dimension name for the columns axis. `None` defaults to
    /// `"column"` and is not written to the track attribute. Ignored when
    /// `columns` is `None`. Cohort tracks should set this to `"sample"`;
    /// feature pileups to `"feature"`; etc.
    pub column_dim_name: Option<String>,
    pub description: Option<String>,
    pub source: Option<String>,
    /// Tool-specific metadata. Stored in the `perbase_zarr_track` attribute
    /// alongside standard fields. Use a namespaced key (e.g. `"clam"`) to
    /// avoid collisions with other tools.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
            column_chunk_size: DEFAULT_COLUMN_CHUNK_SIZE,
            column_dim_name: None,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Metadata read back from an existing track.
#[derive(Debug, Clone)]
pub struct TrackMetadata {
    pub layout: String,
    pub dtype: String,
    pub chunk_size: u64,
    pub column_chunk_size: Option<u64>,
    /// Resolved dimension name for the columns axis. `Some(...)` if the
    /// track has columns (defaulting to `"column"` when the attr was absent);
    /// `None` for 1D tracks.
    pub column_dim_name: Option<String>,
    pub description: Option<String>,
    pub source: Option<String>,
    /// Tool-specific metadata preserved from the `perbase_zarr_track` attribute.
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A PBZ track — a named collection of per-contig data arrays.
///
/// Wraps a zarr group under `/tracks/{name}`. Created via
/// [`Track::create`] or opened via [`Track::open`].
pub struct Track {
    store: ReadableWritableListableStorage,
    name: String,
    group_path: String,
    metadata: TrackMetadata,
    columns: Option<Vec<String>>,
    contig_lengths: HashMap<String, u64>,
}

impl std::fmt::Debug for Track {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Track")
            .field("name", &self.name)
            .field("metadata", &self.metadata)
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

impl Track {
    /// Create a new track in the store.
    ///
    /// Creates the track group with metadata, an optional columns array,
    /// and one data array per contig with appropriate dtype, shape, chunks,
    /// and compression codecs.
    pub fn create(
        store: ReadableWritableListableStorage,
        name: &str,
        config: &TrackConfig,
        contig_lengths: &HashMap<String, u64>,
    ) -> Result<Self> {
        // pbz[impl per_base.data.single-column-1d]
        if let Some(ref s) = config.columns {
            if s.len() == 1 {
                return Err(PbzError::SingleColumnMustBe1D);
            }
            // pbz[impl per_base.columns.unique]
            let mut seen = std::collections::HashSet::new();
            for name in s {
                if !seen.insert(name.as_str()) {
                    return Err(PbzError::Metadata(format!(
                        "duplicate column id {name:?} in track columns"
                    )));
                }
            }
        }

        // pbz[impl per_base.attrs.dtype]
        let zarr_dtype = dtype_to_zarr(&config.dtype)?;
        // pbz[impl missing.fill_value]
        // pbz[impl missing.fill_value.defaults]
        let fill_value = fill_value_for_dtype(&config.dtype)?;
        let has_columns = config.columns.is_some();
        let group_path = format!("/tracks/{name}");
        let num_columns = config.columns.as_ref().map(|c| c.len() as u64).unwrap_or(0);

        // Cap column chunk size so chunks are always full along the columns axis.
        // pbz[impl per_base.attrs.column_chunk_size]
        let actual_column_chunk = if has_columns {
            config.column_chunk_size.min(num_columns)
        } else {
            config.column_chunk_size
        };

        // Build track metadata attribute: standard fields + extra
        // pbz[impl track.attrs.namespace]
        // pbz[impl per_base.attrs.namespace]
        // pbz[impl track.attrs.layout]
        // pbz[impl per_base.attrs.chunk_size]
        let mut track_meta = serde_json::json!({
            "layout": PER_BASE_LAYOUT,
            "dtype": config.dtype,
            "chunk_size": config.chunk_size,
        });
        if has_columns {
            track_meta["column_chunk_size"] = serde_json::json!(actual_column_chunk);
            // pbz[impl per_base.attrs.column_dim_name]
            if let Some(ref dim) = config.column_dim_name {
                if dim.is_empty() {
                    return Err(PbzError::Metadata(
                        "column_dim_name must be a non-empty string".into(),
                    ));
                }
                track_meta["column_dim_name"] = serde_json::json!(dim);
            }
        }
        if let Some(ref desc) = config.description {
            track_meta["description"] = serde_json::json!(desc);
        }
        if let Some(ref src) = config.source {
            track_meta["source"] = serde_json::json!(src);
        }
        // pbz[impl track.attrs.preserve-unknown]
        // pbz[impl reserved.namespace.perbase_zarr]
        // pbz[impl reserved.preserve-unknown]
        // pbz[impl forward.preserve-roundtrip]
        if let serde_json::Value::Object(ref mut map) = track_meta {
            for (k, v) in &config.extra {
                map.insert(k.clone(), v.clone());
            }
        }

        let mut group_attrs = serde_json::Map::new();
        group_attrs.insert(TRACK_ATTR_KEY.into(), track_meta);

        let group = GroupBuilder::new()
            .attributes(group_attrs)
            .build(store.clone(), &group_path)
            .map_err(|e| PbzError::Store(e.to_string()))?;
        group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let effective_dim_name = config
            .column_dim_name
            .as_deref()
            .unwrap_or(DEFAULT_COLUMN_DIM_NAME);

        // Columns array (if multi-column)
        // pbz[impl per_base.columns.array]
        // pbz[impl per_base.columns.order]
        if let Some(ref cols) = config.columns {
            let columns_path = format!("{group_path}/columns");
            let columns_array = ArrayBuilder::new(
                vec![num_columns],
                vec![num_columns],
                data_type::string(),
                "",
            )
            // The columns array's own dimension name mirrors the second axis name.
            .dimension_names(Some([effective_dim_name]))
            .build(store.clone(), &columns_path)
            .map_err(|e| PbzError::Store(e.to_string()))?;
            columns_array
                .store_metadata()
                .map_err(|e| PbzError::Store(e.to_string()))?;
            columns_array
                .store_chunk(&[0], cols.clone())
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        // Per-contig data arrays
        // pbz[impl per_base.data.location]
        // pbz[impl per_base.data.dtype-match]
        for (contig, &length) in contig_lengths {
            let array_path = format!("{group_path}/{contig}");

            let (shape, chunks) = if has_columns {
                // pbz[impl per_base.data.shape-2d]
                (
                    vec![length, num_columns],
                    vec![config.chunk_size, actual_column_chunk],
                )
            } else {
                // pbz[impl per_base.data.shape-1d]
                (vec![length], vec![config.chunk_size])
            };

            let mut builder =
                ArrayBuilder::new(shape, chunks, zarr_dtype.clone(), fill_value.clone());

            // pbz[impl compression.codec-agnostic]
            if config.dtype == "bool" {
                builder.array_to_bytes_codec(Arc::new(PackBitsCodec::default()));
            } else {
                let typesize = dtype_byte_size(&config.dtype);
                // pbz[impl compression.default]
                builder.bytes_to_bytes_codecs(vec![Arc::new(
                    BloscCodec::new(
                        BloscCompressor::Zstd,
                        BloscCompressionLevel::try_from(5)
                            .map_err(|e| PbzError::Store(e.to_string()))?,
                        None,
                        BloscShuffleMode::Shuffle,
                        Some(typesize),
                    )
                    .map_err(|e| PbzError::Store(e.to_string()))?,
                )]);
            }

            // pbz[impl per_base.data.dim-names]
            // pbz[impl per_base.attrs.column_dim_name]
            let dim_names: Vec<&str> = if has_columns {
                vec!["position", effective_dim_name]
            } else {
                vec!["position"]
            };
            builder.dimension_names(Some(dim_names));

            let array = builder
                .build(store.clone(), &array_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
            array
                .store_metadata()
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        let metadata = TrackMetadata {
            layout: PER_BASE_LAYOUT.to_string(),
            dtype: config.dtype.clone(),
            chunk_size: config.chunk_size,
            column_chunk_size: if has_columns {
                Some(actual_column_chunk)
            } else {
                None
            },
            column_dim_name: if has_columns {
                Some(effective_dim_name.to_string())
            } else {
                None
            },
            description: config.description.clone(),
            source: config.source.clone(),
            extra: config.extra.clone(),
        };

        Ok(Self {
            store,
            name: name.to_string(),
            group_path,
            metadata,
            columns: config.columns.clone(),
            contig_lengths: contig_lengths.clone(),
        })
    }

    /// Open an existing track from the store.
    ///
    /// Reads the track group metadata and columns array (if columnar).
    /// Unrecognized keys in the `perbase_zarr_track` attribute are preserved
    /// in [`TrackMetadata::extra`].
    pub fn open(
        store: ReadableWritableListableStorage,
        name: &str,
        contig_lengths: &HashMap<String, u64>,
    ) -> Result<Self> {
        let group_path = format!("/tracks/{name}");

        let group =
            Group::open(store.clone(), &group_path).map_err(|e| PbzError::Store(e.to_string()))?;

        let track_meta = group
            .attributes()
            .get(TRACK_ATTR_KEY)
            .ok_or_else(|| {
                PbzError::Metadata(format!(
                    "group '{group_path}' missing '{TRACK_ATTR_KEY}' attribute"
                ))
            })?
            .clone();

        let track_obj = track_meta
            .as_object()
            .ok_or_else(|| PbzError::Metadata("track metadata is not an object".into()))?;

        // pbz[impl track.attrs.layout]
        let layout = track_obj
            .get("layout")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PbzError::MissingLayout {
                track: name.to_string(),
            })?
            .to_string();
        if layout != PER_BASE_LAYOUT {
            return Err(PbzError::UnknownLayout {
                track: name.to_string(),
                layout,
            });
        }

        let dtype = track_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PbzError::Metadata("track metadata missing 'dtype'".into()))?
            .to_string();

        let chunk_size = track_obj
            .get("chunk_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| PbzError::Metadata("track metadata missing 'chunk_size'".into()))?;

        // Authoritative: presence of the columns array determines multi-column.
        let columns_path = format!("{group_path}/columns");
        let has_columns = Array::open(store.clone(), &columns_path).is_ok();

        let column_chunk_size = if has_columns {
            Some(
                track_obj
                    .get("column_chunk_size")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        PbzError::Metadata("multi-column track missing 'column_chunk_size'".into())
                    })?,
            )
        } else {
            None
        };

        let column_dim_name = if has_columns {
            Some(
                track_obj
                    .get("column_dim_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(DEFAULT_COLUMN_DIM_NAME)
                    .to_string(),
            )
        } else {
            None
        };

        let description = track_obj
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);

        let source = track_obj
            .get("source")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Collect unrecognized keys into extra
        // pbz[impl track.attrs.preserve-unknown]
        let mut extra = serde_json::Map::new();
        for (k, v) in track_obj {
            if !STANDARD_FIELDS.contains(&k.as_str()) {
                extra.insert(k.clone(), v.clone());
            }
        }

        let columns = if has_columns {
            let columns_array = Array::open(store.clone(), &columns_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
            let cols: Vec<String> = columns_array
                .retrieve_chunk::<Vec<String>>(&[0])
                .map_err(|e| PbzError::Store(e.to_string()))?;
            Some(cols)
        } else {
            None
        };

        let metadata = TrackMetadata {
            layout: PER_BASE_LAYOUT.to_string(),
            dtype,
            chunk_size,
            column_chunk_size,
            column_dim_name,
            description,
            source,
            extra,
        };

        Ok(Self {
            store,
            name: name.to_string(),
            group_path,
            metadata,
            columns,
            contig_lengths: contig_lengths.clone(),
        })
    }

    // -- Accessors ----------------------------------------------------------

    /// Track name (relative to `/tracks/`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Track metadata.
    pub fn metadata(&self) -> &TrackMetadata {
        &self.metadata
    }

    /// Update the `description` field of the track's metadata, preserving all
    /// other fields including the `extra` namespaced map.
    ///
    /// `None` clears the field.
    pub fn set_description(&mut self, description: Option<&str>) -> Result<()> {
        self.update_metadata_field("description", description.map(serde_json::Value::from))?;
        self.metadata.description = description.map(|s| s.to_string());
        Ok(())
    }

    /// Update the `source` field of the track's metadata, preserving all other
    /// fields including the `extra` namespaced map.
    ///
    /// `None` clears the field.
    pub fn set_source(&mut self, source: Option<&str>) -> Result<()> {
        self.update_metadata_field("source", source.map(serde_json::Value::from))?;
        self.metadata.source = source.map(|s| s.to_string());
        Ok(())
    }

    fn update_metadata_field(&self, field: &str, value: Option<serde_json::Value>) -> Result<()> {
        // Mutate the existing group's attributes in place so additional_fields
        // (set by other tools or future extensions) survive the rewrite.
        // Rebuilding via GroupBuilder would silently drop them.
        let mut group = Group::open(self.store.clone(), &self.group_path)
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let track_meta = group
            .attributes_mut()
            .get_mut(TRACK_ATTR_KEY)
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| {
                PbzError::Metadata(format!(
                    "track group '{}' missing '{TRACK_ATTR_KEY}' attribute",
                    self.group_path
                ))
            })?;
        match value {
            Some(v) => {
                track_meta.insert(field.to_string(), v);
            }
            None => {
                track_meta.remove(field);
            }
        }
        group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Ok(())
    }

    /// Whether this track has a columns dimension.
    pub fn has_columns(&self) -> bool {
        self.columns.is_some()
    }

    /// Column names, if this is a multi-column track.
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
    }

    /// Xarray dimension name for the columns axis.
    ///
    /// Returns the stored `column_dim_name` attribute, or `"column"` if the
    /// track is 1D or the attribute was absent (i.e. the default).
    pub fn column_dim_name(&self) -> &str {
        self.metadata
            .column_dim_name
            .as_deref()
            .unwrap_or(DEFAULT_COLUMN_DIM_NAME)
    }

    /// The track layout (always `"per_base"` for spec v0.1).
    pub fn layout(&self) -> &str {
        &self.metadata.layout
    }

    /// The dtype string (e.g. "uint32", "bool").
    pub fn dtype(&self) -> &str {
        &self.metadata.dtype
    }

    /// Chunk size along the position axis.
    pub fn chunk_size(&self) -> u64 {
        self.metadata.chunk_size
    }

    /// Escape hatch: open the raw zarrs Array for a contig.
    pub fn zarr_array(
        &self,
        contig: &str,
    ) -> Result<Array<dyn zarrs::storage::ReadableWritableListableStorageTraits>> {
        self.validate_contig(contig)?;
        let path = format!("{}/{contig}", self.group_path);
        Array::open(self.store.clone(), &path).map_err(|e| PbzError::Store(e.to_string()))
    }

    // -- Chunk math ---------------------------------------------------------

    // pbz[impl coords.zero-based-half-open]
    /// Convert a 0-based position to a chunk index.
    pub fn position_to_chunk(&self, position: u64) -> u64 {
        position / self.metadata.chunk_size
    }

    /// Get the [start, end) bounds of a chunk, clamped to contig length.
    pub fn chunk_bounds(&self, chunk_idx: u64, contig_length: u64) -> (u64, u64) {
        let start = chunk_idx * self.metadata.chunk_size;
        let end = ((chunk_idx + 1) * self.metadata.chunk_size).min(contig_length);
        (start, end)
    }

    /// Get the range of chunk indices that overlap a half-open region [start, end).
    pub fn overlapping_chunks(&self, start: u64, end: u64) -> Range<u64> {
        let start_chunk = self.position_to_chunk(start);
        let end_chunk = self.position_to_chunk(end.saturating_sub(1)) + 1;
        start_chunk..end_chunk
    }

    // -- Columnar chunk I/O -------------------------------------------------

    /// Write a 2D chunk to a columnar track.
    ///
    /// `data` shape must be `(positions, num_columns)`. Full chunks must have
    /// exactly `chunk_size` rows. Partial chunks are allowed only at the last
    /// chunk position for the contig.
    pub fn write_chunk<T: ElementOwned>(
        &self,
        contig: &str,
        chunk_idx: u64,
        data: Array2<T>,
    ) -> Result<()> {
        if !self.has_columns() {
            return Err(PbzError::Metadata(
                "write_chunk called on a scalar track; use write_chunk_1d instead".into(),
            ));
        }

        let expected_cols = self.columns.as_ref().map(|c| c.len()).unwrap_or(0);
        if data.ncols() != expected_cols {
            return Err(PbzError::Metadata(format!(
                "column count mismatch: data has {} columns but track has {}",
                data.ncols(),
                expected_cols
            )));
        }

        let contig_length = self.contig_length(contig)?;
        let array = self.zarr_array(contig)?;
        let (chunk_start, chunk_end) = self.chunk_bounds(chunk_idx, contig_length);
        let expected_rows = (chunk_end - chunk_start) as usize;

        if data.nrows() == self.metadata.chunk_size as usize {
            array
                .store_chunk(&[chunk_idx, 0], data)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        } else if data.nrows() == expected_rows && chunk_end == contig_length {
            let subset =
                ArraySubset::new_with_ranges(&[0..data.nrows() as u64, 0..data.ncols() as u64]);
            array
                .store_chunk_subset(&[chunk_idx, 0], &subset, data)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        } else {
            return Err(PbzError::Metadata(format!(
                "chunk {} has {} rows but expected {} (full) or {} (last chunk)",
                chunk_idx,
                data.nrows(),
                self.metadata.chunk_size,
                expected_rows
            )));
        }

        Ok(())
    }

    /// Read a 2D chunk from a columnar track.
    ///
    /// Returns shape `(chunk_positions, num_columns)`. The last chunk may
    /// have fewer rows than `chunk_size`.
    pub fn read_chunk<T: ElementOwned>(&self, contig: &str, chunk_idx: u64) -> Result<Array2<T>> {
        if !self.has_columns() {
            return Err(PbzError::Metadata(
                "read_chunk called on a scalar track; use read_chunk_1d instead".into(),
            ));
        }

        let contig_length = self.contig_length(contig)?;
        let num_columns = self.columns.as_ref().map(|c| c.len() as u64).unwrap_or(0);
        let (start, end) = self.chunk_bounds(chunk_idx, contig_length);
        let subset = ArraySubset::new_with_ranges(&[start..end, 0..num_columns]);

        let array = self.zarr_array(contig)?;
        let data: ArrayD<T> = array
            .retrieve_array_subset(&subset)
            .map_err(|e| PbzError::Store(e.to_string()))?;

        data.into_dimensionality()
            .map_err(|e| PbzError::Store(e.to_string()))
    }

    // -- Scalar chunk I/O ---------------------------------------------------

    /// Write a 1D chunk to a scalar (non-columnar) track.
    pub fn write_chunk_1d<T: ElementOwned>(
        &self,
        contig: &str,
        chunk_idx: u64,
        data: Array1<T>,
    ) -> Result<()> {
        if self.has_columns() {
            return Err(PbzError::Metadata(
                "write_chunk_1d called on a columnar track; use write_chunk instead".into(),
            ));
        }

        let contig_length = self.contig_length(contig)?;
        let array = self.zarr_array(contig)?;
        let (chunk_start, chunk_end) = self.chunk_bounds(chunk_idx, contig_length);
        let expected_rows = (chunk_end - chunk_start) as usize;

        if data.len() == self.metadata.chunk_size as usize {
            array
                .store_chunk(&[chunk_idx], data)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        } else if data.len() == expected_rows && chunk_end == contig_length {
            #[allow(clippy::single_range_in_vec_init)]
            let subset = ArraySubset::new_with_ranges(&[0..data.len() as u64]);
            array
                .store_chunk_subset(&[chunk_idx], &subset, data)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        } else {
            return Err(PbzError::Metadata(format!(
                "chunk {} has {} elements but expected {} or {}",
                chunk_idx,
                data.len(),
                self.metadata.chunk_size,
                expected_rows
            )));
        }

        Ok(())
    }

    /// Read a 1D chunk from a scalar (non-columnar) track.
    pub fn read_chunk_1d<T: ElementOwned>(
        &self,
        contig: &str,
        chunk_idx: u64,
    ) -> Result<Array1<T>> {
        if self.has_columns() {
            return Err(PbzError::Metadata(
                "read_chunk_1d called on a columnar track; use read_chunk instead".into(),
            ));
        }

        let contig_length = self.contig_length(contig)?;
        let (start, end) = self.chunk_bounds(chunk_idx, contig_length);
        #[allow(clippy::single_range_in_vec_init)]
        let subset = ArraySubset::new_with_ranges(&[start..end]);

        let array = self.zarr_array(contig)?;
        let data: ArrayD<T> = array
            .retrieve_array_subset(&subset)
            .map_err(|e| PbzError::Store(e.to_string()))?;

        data.into_dimensionality()
            .map_err(|e| PbzError::Store(e.to_string()))
    }

    // -- Private helpers ----------------------------------------------------

    fn validate_contig(&self, contig: &str) -> Result<()> {
        if self.contig_lengths.contains_key(contig) {
            Ok(())
        } else {
            Err(PbzError::ContigNotFound {
                contig: contig.into(),
                available: self.contig_lengths.keys().cloned().collect(),
            })
        }
    }

    fn contig_length(&self, contig: &str) -> Result<u64> {
        self.contig_lengths
            .get(contig)
            .copied()
            .ok_or_else(|| PbzError::ContigNotFound {
                contig: contig.into(),
                available: self.contig_lengths.keys().cloned().collect(),
            })
    }
}

// -- Dtype helpers ----------------------------------------------------------

fn dtype_to_zarr(dtype: &str) -> Result<DataType> {
    match dtype {
        "uint8" => Ok(data_type::uint8()),
        "uint16" => Ok(data_type::uint16()),
        "uint32" => Ok(data_type::uint32()),
        "int8" => Ok(data_type::int8()),
        "int16" => Ok(data_type::int16()),
        "int32" => Ok(data_type::int32()),
        "float32" => Ok(data_type::float32()),
        "float64" => Ok(data_type::float64()),
        "bool" => Ok(data_type::bool()),
        other => Err(PbzError::InvalidDtype {
            dtype: other.into(),
        }),
    }
}

fn fill_value_for_dtype(dtype: &str) -> Result<FillValue> {
    match dtype {
        "uint8" => Ok(FillValue::from(0u8)),
        "uint16" => Ok(FillValue::from(0u16)),
        "uint32" => Ok(FillValue::from(0u32)),
        "int8" => Ok(FillValue::from(0i8)),
        "int16" => Ok(FillValue::from(0i16)),
        "int32" => Ok(FillValue::from(0i32)),
        "float32" => Ok(FillValue::from(f32::NAN)),
        "float64" => Ok(FillValue::from(f64::NAN)),
        "bool" => Ok(FillValue::from(false)),
        other => Err(PbzError::InvalidDtype {
            dtype: other.into(),
        }),
    }
}

fn dtype_byte_size(dtype: &str) -> usize {
    match dtype {
        "uint8" | "int8" | "bool" => 1,
        "uint16" | "int16" => 2,
        "uint32" | "int32" | "float32" => 4,
        "float64" => 8,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PbzStore;
    use tempfile::TempDir;

    fn test_store() -> (TempDir, PbzStore) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.pbz.zarr");
        let contigs = vec!["chr1".to_string(), "chr2".to_string()];
        let lengths = vec![5000u64, 3000];
        let store = PbzStore::create(&path, &contigs, &lengths).unwrap();
        (dir, store)
    }

    // -- Dtype helpers ------------------------------------------------------

    #[test]
    fn dtype_to_zarr_valid() {
        assert_eq!(dtype_to_zarr("uint32").unwrap(), data_type::uint32());
        assert_eq!(dtype_to_zarr("bool").unwrap(), data_type::bool());
        assert_eq!(dtype_to_zarr("float64").unwrap(), data_type::float64());
    }

    #[test]
    fn dtype_to_zarr_invalid() {
        let err = dtype_to_zarr("complex128").unwrap_err();
        assert!(matches!(err, PbzError::InvalidDtype { .. }));
    }

    // pbz[verify missing.fill_value.defaults]
    #[test]
    fn fill_values() {
        assert_eq!(
            fill_value_for_dtype("uint32").unwrap(),
            FillValue::from(0u32)
        );
        assert_eq!(
            fill_value_for_dtype("bool").unwrap(),
            FillValue::from(false)
        );
        fill_value_for_dtype("float32").unwrap();
    }

    #[test]
    fn track_config_defaults() {
        let config = TrackConfig::default();
        assert_eq!(config.dtype, "uint32");
        assert_eq!(config.chunk_size, 1_000_000);
        assert_eq!(config.column_chunk_size, 16);
        assert!(config.columns.is_none());
        assert!(config.extra.is_empty());
    }

    // -- Track::create ------------------------------------------------------

    #[test]
    fn create_columnar_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into(), "s3".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        assert_eq!(track.name(), "depths");
        assert_eq!(track.metadata().dtype, "uint32");
        assert!(track.has_columns());
        assert_eq!(track.columns().unwrap(), &["s1", "s2", "s3"]);
    }

    #[test]
    fn create_scalar_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "bool".into(),
            columns: None,
            chunk_size: 1000,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "mask",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        assert!(!track.has_columns());
        assert!(track.columns().is_none());
        assert_eq!(track.metadata().dtype, "bool");
    }

    // -- Track::open --------------------------------------------------------

    #[test]
    fn open_columnar_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            description: Some("test track".into()),
            source: Some("unit test".into()),
            ..Default::default()
        };
        Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let track = Track::open(store.storage().clone(), "depths", store.contig_lengths()).unwrap();

        assert_eq!(track.name(), "depths");
        assert_eq!(track.metadata().dtype, "uint32");
        assert_eq!(track.metadata().chunk_size, 1000);
        assert_eq!(track.metadata().column_chunk_size, Some(2));
        assert!(track.has_columns());
        assert_eq!(track.columns().unwrap(), &["s1", "s2"]);
        assert_eq!(track.metadata().description.as_deref(), Some("test track"));
        assert_eq!(track.metadata().source.as_deref(), Some("unit test"));
    }

    #[test]
    fn open_scalar_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "float32".into(),
            columns: None,
            chunk_size: 500,
            ..Default::default()
        };
        Track::create(
            store.storage().clone(),
            "signal",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let track = Track::open(store.storage().clone(), "signal", store.contig_lengths()).unwrap();

        assert!(!track.has_columns());
        assert!(track.columns().is_none());
        assert_eq!(track.metadata().chunk_size, 500);
        assert!(track.metadata().column_chunk_size.is_none());
    }

    #[test]
    fn extra_metadata_round_trip() {
        let (_dir, store) = test_store();
        let mut extra = serde_json::Map::new();
        extra.insert(
            "clam".into(),
            serde_json::json!({
                "callable_loci_type": "sample_masks",
                "populations": [{"name": "pop1", "samples": ["s1", "s2"]}]
            }),
        );

        let config = TrackConfig {
            dtype: "bool".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            extra,
            ..Default::default()
        };
        Track::create(
            store.storage().clone(),
            "mask",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let track = Track::open(store.storage().clone(), "mask", store.contig_lengths()).unwrap();

        let clam_meta = track.metadata().extra.get("clam").unwrap();
        assert_eq!(clam_meta["callable_loci_type"], "sample_masks");
        assert_eq!(clam_meta["populations"][0]["name"], "pop1");
    }

    // -- Chunk math ---------------------------------------------------------

    #[test]
    fn chunk_math() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            chunk_size: 1000,
            columns: Some(vec!["s1".into(), "s2".into()]),
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "test",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        assert_eq!(track.position_to_chunk(0), 0);
        assert_eq!(track.position_to_chunk(999), 0);
        assert_eq!(track.position_to_chunk(1000), 1);
        assert_eq!(track.position_to_chunk(4999), 4);

        assert_eq!(track.chunk_bounds(0, 5000), (0, 1000));
        assert_eq!(track.chunk_bounds(4, 5000), (4000, 5000));
        assert_eq!(track.chunk_bounds(2, 3000), (2000, 3000));

        assert_eq!(track.overlapping_chunks(0, 1000), 0..1);
        assert_eq!(track.overlapping_chunks(500, 2500), 0..3);
        assert_eq!(track.overlapping_chunks(0, 5000), 0..5);
        assert_eq!(track.overlapping_chunks(1000, 1001), 1..2);
    }

    // -- Columnar chunk I/O -------------------------------------------------

    #[test]
    fn write_read_chunk_columnar() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let mut data = Array2::<u32>::zeros((1000, 2));
        for i in 0..1000 {
            data[[i, 0]] = 10;
            data[[i, 1]] = 20;
        }

        track.write_chunk("chr1", 0, data).unwrap();
        let read_back: Array2<u32> = track.read_chunk("chr1", 0).unwrap();

        assert_eq!(read_back.shape(), &[1000, 2]);
        assert_eq!(read_back[[0, 0]], 10);
        assert_eq!(read_back[[999, 1]], 20);
    }

    #[test]
    fn write_read_partial_last_chunk() {
        let (_dir, store) = test_store();
        // chr2 is 3000bp, chunk_size=1100 → 2 full chunks + partial (800 rows)
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1100,
            column_chunk_size: 16,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let data = Array2::<u32>::from_elem((800, 2), 42);
        track.write_chunk("chr2", 2, data).unwrap();

        let read_back: Array2<u32> = track.read_chunk("chr2", 2).unwrap();
        assert_eq!(read_back.shape(), &[800, 2]);
        assert_eq!(read_back[[0, 0]], 42);
        assert_eq!(read_back[[799, 0]], 42);
    }

    #[test]
    fn write_read_bool_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "bool".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "mask",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let mut data = Array2::<bool>::from_elem((1000, 2), false);
        for i in 0..1000 {
            data[[i, 0]] = true;
            data[[i, 1]] = i % 2 == 0;
        }

        track.write_chunk("chr1", 0, data.clone()).unwrap();
        let read_back: Array2<bool> = track.read_chunk("chr1", 0).unwrap();

        assert_eq!(read_back.shape(), &[1000, 2]);
        for i in 0..1000 {
            assert_eq!(read_back[[i, 0]], true);
            assert_eq!(read_back[[i, 1]], i % 2 == 0);
        }
    }

    #[test]
    fn write_chunk_column_mismatch() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let data = Array2::<u32>::zeros((1000, 3));
        let err = track.write_chunk("chr1", 0, data).unwrap_err();
        assert!(matches!(err, PbzError::Metadata(_)));
    }

    // -- Scalar chunk I/O ---------------------------------------------------

    #[test]
    fn write_read_scalar_track() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "float32".into(),
            columns: None,
            chunk_size: 1000,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "signal",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let data = Array1::<f32>::from_vec((0..1000).map(|i| i as f32 * 0.1).collect());
        track.write_chunk_1d("chr1", 0, data).unwrap();

        let read_back: Array1<f32> = track.read_chunk_1d("chr1", 0).unwrap();
        assert_eq!(read_back.len(), 1000);
        assert!((read_back[0] - 0.0).abs() < 1e-6);
        assert!((read_back[999] - 99.9).abs() < 0.01);
    }

    #[test]
    fn write_read_scalar_bool_partial() {
        let (_dir, store) = test_store();
        // chr2 is 3000bp, chunk_size=1000 → 3 full chunks
        // Use chunk_size=1100 for partial: 3000/1100 = 2 full + 800 partial
        let config = TrackConfig {
            dtype: "bool".into(),
            columns: None,
            chunk_size: 1100,
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "mask",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let data = Array1::<bool>::from_vec((0..800).map(|i| i % 2 == 0).collect());
        track.write_chunk_1d("chr2", 2, data).unwrap();

        let read_back: Array1<bool> = track.read_chunk_1d("chr2", 2).unwrap();
        assert_eq!(read_back.len(), 800);
        assert_eq!(read_back[0], true);
        assert_eq!(read_back[1], false);
        assert_eq!(read_back[799], false);
    }

    #[test]
    fn dimensionality_mismatch_errors() {
        let (_dir, store) = test_store();

        // Columnar track
        let col_track = Track::create(
            store.storage().clone(),
            "col",
            &TrackConfig {
                dtype: "uint32".into(),
                columns: Some(vec!["s1".into(), "s2".into()]),
                chunk_size: 1000,
                ..Default::default()
            },
            store.contig_lengths(),
        )
        .unwrap();

        // Scalar track
        let scalar_track = Track::create(
            store.storage().clone(),
            "scalar",
            &TrackConfig {
                dtype: "uint32".into(),
                columns: None,
                chunk_size: 1000,
                ..Default::default()
            },
            store.contig_lengths(),
        )
        .unwrap();

        // 1D methods on columnar track → error
        let data_1d = Array1::<u32>::zeros(1000);
        assert!(col_track.write_chunk_1d("chr1", 0, data_1d).is_err());
        assert!(col_track.read_chunk_1d::<u32>("chr1", 0).is_err());

        // 2D methods on scalar track → error
        let data_2d = Array2::<u32>::zeros((1000, 1));
        assert!(scalar_track.write_chunk("chr1", 0, data_2d).is_err());
        assert!(scalar_track.read_chunk::<u32>("chr1", 0).is_err());
    }

    #[test]
    fn set_description_updates_attribute() {
        use crate::store::PbzStore;
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[1000u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 100,
            column_chunk_size: 1,
            description: Some("orig".into()),
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        let mut track = store.create_track("t", cfg).unwrap();
        track.set_description(Some("updated")).unwrap();

        drop(track);
        drop(store);
        let store2 = PbzStore::open(&store_path).unwrap();
        let track2 = store2.track("t").unwrap();
        assert_eq!(track2.metadata().description.as_deref(), Some("updated"));
    }

    #[test]
    fn set_description_clear_with_none() {
        use crate::store::PbzStore;
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: Some("orig".into()),
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        let mut track = store.create_track("t", cfg).unwrap();
        track.set_description(None).unwrap();
        drop(track);
        drop(store);
        let store2 = PbzStore::open(&store_path).unwrap();
        let track2 = store2.track("t").unwrap();
        assert_eq!(track2.metadata().description, None);
    }

    #[test]
    fn set_source_updates_attribute() {
        use crate::store::PbzStore;
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[1000u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: Some("orig".into()),
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        let mut track = store.create_track("t", cfg).unwrap();
        track.set_source(Some("updated")).unwrap();

        drop(track);
        drop(store);
        let store2 = PbzStore::open(&store_path).unwrap();
        let track2 = store2.track("t").unwrap();
        assert_eq!(track2.metadata().source.as_deref(), Some("updated"));
    }

    #[test]
    fn set_description_preserves_extra_metadata() {
        use crate::store::PbzStore;
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();

        let mut extra = serde_json::Map::new();
        extra.insert(
            "clam".into(),
            serde_json::json!({"populations": [{"name": "p1", "samples": ["s1"]}]}),
        );
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: Some("orig".into()),
            source: None,
            extra,
            column_dim_name: None,
        };
        let mut track = store.create_track("t", cfg).unwrap();
        track.set_description(Some("updated")).unwrap();
        drop(track);
        drop(store);

        let store2 = PbzStore::open(&store_path).unwrap();
        let track2 = store2.track("t").unwrap();
        assert_eq!(track2.metadata().description.as_deref(), Some("updated"));
        let clam = track2.metadata().extra.get("clam").unwrap();
        assert_eq!(clam["populations"][0]["name"], "p1");
    }

    #[test]
    fn set_description_preserves_additional_fields() {
        // Regression: update_metadata_field used to rebuild the group via
        // GroupBuilder, which silently dropped any GroupMetadataV3 additional_fields
        // (top-level JSON keys outside `attributes`). Mutating in-place via
        // attributes_mut() preserves them. Exercise that here by injecting a
        // non-empty additional_fields on the track group, then calling
        // set_description and asserting the field survives the round-trip.
        use crate::store::PbzStore;
        use zarrs::group::GroupBuilder;
        use zarrs::metadata::GroupMetadata;
        use zarrs::metadata::v3::{AdditionalFieldV3, AdditionalFieldsV3};

        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: Some("orig".into()),
            source: None,
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        store.create_track("t", cfg).unwrap();

        // Rewrite /tracks/t with the same attributes plus a custom
        // additional_fields entry. must_understand=false so opening the group
        // doesn't reject it.
        let group_path = "/tracks/t";
        let original_attrs = Group::open(store.storage().clone(), group_path)
            .unwrap()
            .attributes()
            .clone();

        let mut additional = AdditionalFieldsV3::new();
        additional.insert(
            "external_tool_marker".to_string(),
            AdditionalFieldV3::new(serde_json::json!({"tag": "preserve_me"}), false),
        );

        let group = GroupBuilder::new()
            .attributes(original_attrs)
            .additional_fields(additional)
            .build(store.storage().clone(), group_path)
            .unwrap();
        group.store_metadata().unwrap();

        // Sanity: the field is on disk before set_description runs.
        {
            let g = Group::open(store.storage().clone(), group_path).unwrap();
            if let GroupMetadata::V3(meta) = g.metadata() {
                assert!(
                    meta.additional_fields.contains_key("external_tool_marker"),
                    "precondition failed: additional_fields not persisted"
                );
            } else {
                panic!("expected V3 group metadata");
            }
        }

        drop(store);
        let store2 = PbzStore::open(&store_path).unwrap();
        let mut track = store2.track("t").unwrap();
        track.set_description(Some("changed")).unwrap();
        drop(track);
        drop(store2);

        let store3 = PbzStore::open(&store_path).unwrap();
        let track3 = store3.track("t").unwrap();
        assert_eq!(track3.metadata().description.as_deref(), Some("changed"));

        let g = Group::open(store3.storage().clone(), group_path).unwrap();
        match g.metadata() {
            GroupMetadata::V3(meta) => {
                let field = meta
                    .additional_fields
                    .get("external_tool_marker")
                    .expect("additional_fields entry was dropped by set_description");
                assert_eq!(field.as_value()["tag"], "preserve_me");
            }
            _ => panic!("expected V3 group metadata"),
        }
    }

    #[test]
    fn set_source_clear_with_none() {
        use crate::store::PbzStore;
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().join("s.pbz.zarr");
        let store = PbzStore::create(&store_path, &["chr1".to_string()], &[100u64]).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: Some("orig".into()),
            extra: serde_json::Map::new(),
            column_dim_name: None,
        };
        let mut track = store.create_track("t", cfg).unwrap();
        track.set_source(None).unwrap();
        drop(track);
        drop(store);
        let store2 = PbzStore::open(&store_path).unwrap();
        let track2 = store2.track("t").unwrap();
        assert_eq!(track2.metadata().source, None);
    }

    #[test]
    fn create_with_column_dim_name() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint16".into(),
            columns: Some(vec!["m6A_h1".into(), "m6A_h2".into(), "nuc_all".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            column_dim_name: Some("feature".into()),
            ..Default::default()
        };
        let track = Track::create(
            store.storage().clone(),
            "pileup",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        assert!(track.has_columns());
        assert_eq!(track.columns().unwrap(), &["m6A_h1", "m6A_h2", "nuc_all"]);
        assert_eq!(track.column_dim_name(), "feature");

        // Verify the data array's dimension names include "feature" as the column dim.
        let arr = track.zarr_array("chr1").unwrap();
        let dims: Vec<String> = arr
            .dimension_names()
            .as_ref()
            .expect("dimension names should be set")
            .iter()
            .map(|d| d.clone().unwrap_or_default())
            .collect();
        assert_eq!(dims, vec!["position".to_string(), "feature".to_string()]);
    }

    #[test]
    fn open_preserves_column_dim_name() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["s1".into(), "s2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            column_dim_name: Some("sample".into()),
            ..Default::default()
        };
        Track::create(
            store.storage().clone(),
            "depths",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let track = Track::open(store.storage().clone(), "depths", store.contig_lengths()).unwrap();
        assert_eq!(track.column_dim_name(), "sample");
    }

    #[test]
    fn column_dim_name_defaults_to_column_when_absent() {
        let (_dir, store) = test_store();
        let config = TrackConfig {
            dtype: "uint32".into(),
            columns: Some(vec!["c1".into(), "c2".into()]),
            chunk_size: 1000,
            column_chunk_size: 16,
            column_dim_name: None,
            ..Default::default()
        };
        Track::create(
            store.storage().clone(),
            "t",
            &config,
            store.contig_lengths(),
        )
        .unwrap();

        let track = Track::open(store.storage().clone(), "t", store.contig_lengths()).unwrap();
        assert_eq!(track.column_dim_name(), "column");
    }
}
