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
    "dtype",
    "chunk_size",
    "column_chunk_size",
    "has_columns",
    "description",
    "source",
];

/// Default chunk size along the position axis (1 million base pairs).
pub const DEFAULT_CHUNK_SIZE: u64 = 1_000_000;

/// Default chunk size along the column axis.
pub const DEFAULT_COLUMN_CHUNK_SIZE: u64 = 16;

/// Configuration for creating a new track.
pub struct TrackConfig {
    pub dtype: String,
    pub columns: Option<Vec<String>>,
    pub chunk_size: u64,
    pub column_chunk_size: u64,
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
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        }
    }
}

/// Metadata read back from an existing track.
#[derive(Debug, Clone)]
pub struct TrackMetadata {
    pub dtype: String,
    pub chunk_size: u64,
    pub column_chunk_size: Option<u64>,
    pub has_columns: bool,
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
        let zarr_dtype = dtype_to_zarr(&config.dtype)?;
        let fill_value = fill_value_for_dtype(&config.dtype)?;
        let has_columns = config.columns.is_some();
        let group_path = format!("/tracks/{name}");
        let num_columns = config.columns.as_ref().map(|c| c.len() as u64).unwrap_or(0);

        // Cap column chunk size so chunks are always full along the column axis.
        let actual_col_chunk = if has_columns {
            config.column_chunk_size.min(num_columns)
        } else {
            config.column_chunk_size
        };

        // Build track metadata attribute: standard fields + extra
        let mut track_meta = serde_json::json!({
            "dtype": config.dtype,
            "chunk_size": config.chunk_size,
            "has_columns": has_columns,
        });
        if has_columns {
            track_meta["column_chunk_size"] = serde_json::json!(actual_col_chunk);
        }
        if let Some(ref desc) = config.description {
            track_meta["description"] = serde_json::json!(desc);
        }
        if let Some(ref src) = config.source {
            track_meta["source"] = serde_json::json!(src);
        }
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

        // Columns array (if columnar)
        if let Some(ref cols) = config.columns {
            let cols_path = format!("{group_path}/columns");
            let cols_array = ArrayBuilder::new(
                vec![num_columns],
                vec![num_columns],
                data_type::string(),
                "",
            )
            .build(store.clone(), &cols_path)
            .map_err(|e| PbzError::Store(e.to_string()))?;
            cols_array
                .store_metadata()
                .map_err(|e| PbzError::Store(e.to_string()))?;
            cols_array
                .store_chunk(&[0], cols.clone())
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        // Per-contig data arrays
        for (contig, &length) in contig_lengths {
            let array_path = format!("{group_path}/{contig}");

            let (shape, chunks) = if has_columns {
                (
                    vec![length, num_columns],
                    vec![config.chunk_size, actual_col_chunk],
                )
            } else {
                (vec![length], vec![config.chunk_size])
            };

            let mut builder =
                ArrayBuilder::new(shape, chunks, zarr_dtype.clone(), fill_value.clone());

            if config.dtype == "bool" {
                builder.array_to_bytes_codec(Arc::new(PackBitsCodec::default()));
            } else {
                let typesize = dtype_byte_size(&config.dtype);
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

            let dim_names: Vec<&str> = if has_columns {
                vec!["position", "column"]
            } else {
                vec!["position"]
            };
            builder.dimension_names(dim_names.into());

            let array = builder
                .build(store.clone(), &array_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
            array
                .store_metadata()
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        let metadata = TrackMetadata {
            dtype: config.dtype.clone(),
            chunk_size: config.chunk_size,
            column_chunk_size: if has_columns {
                Some(actual_col_chunk)
            } else {
                None
            },
            has_columns,
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

        let dtype = track_obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| PbzError::Metadata("track metadata missing 'dtype'".into()))?
            .to_string();

        let chunk_size = track_obj
            .get("chunk_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| PbzError::Metadata("track metadata missing 'chunk_size'".into()))?;

        let has_columns = track_obj
            .get("has_columns")
            .and_then(|v| v.as_bool())
            .ok_or_else(|| PbzError::Metadata("track metadata missing 'has_columns'".into()))?;

        let column_chunk_size = if has_columns {
            Some(
                track_obj
                    .get("column_chunk_size")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        PbzError::Metadata("columnar track missing 'column_chunk_size'".into())
                    })?,
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
        let mut extra = serde_json::Map::new();
        for (k, v) in track_obj {
            if !STANDARD_FIELDS.contains(&k.as_str()) {
                extra.insert(k.clone(), v.clone());
            }
        }

        let columns = if has_columns {
            let cols_path = format!("{group_path}/columns");
            let cols_array = Array::open(store.clone(), &cols_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
            let cols: Vec<String> = cols_array
                .retrieve_chunk::<Vec<String>>(&[0])
                .map_err(|e| PbzError::Store(e.to_string()))?;
            Some(cols)
        } else {
            None
        };

        let metadata = TrackMetadata {
            dtype,
            chunk_size,
            column_chunk_size,
            has_columns,
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

    /// Whether this track has a column dimension.
    pub fn has_columns(&self) -> bool {
        self.metadata.has_columns
    }

    /// Column names, if this is a columnar track.
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
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
        if !self.metadata.has_columns {
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
        if !self.metadata.has_columns {
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
        if self.metadata.has_columns {
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
        if self.metadata.has_columns {
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
            columns: Some(vec!["s1".into()]),
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
            columns: Some(vec!["s1".into()]),
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

        let data = Array2::<u32>::from_elem((800, 1), 42);
        track.write_chunk("chr2", 2, data).unwrap();

        let read_back: Array2<u32> = track.read_chunk("chr2", 2).unwrap();
        assert_eq!(read_back.shape(), &[800, 1]);
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
                columns: Some(vec!["s1".into()]),
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
}
