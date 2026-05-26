//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::Arc;

use hashbrown::HashMap;
use ndarray::Array1;
use serde_json::{Map, Value, json};
use zarrs::array::Array;
use zarrs::array::FillValue;
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::PbzError;
use crate::genome::{Contig, Genome};
use crate::io::Dtype;
use crate::track::{Track, TrackConfig, TrackMetadata};
use crate::{PBZ_FORMAT_VERSION, Result};

/// A handle to an open PBZ store.
pub struct PbzStore {
    // Kept for future write operations (Task 8+).
    #[allow(dead_code)]
    pub(crate) storage: ReadableWritableListableStorage,
    /// Concrete filesystem store for `Array::open` (needs the concrete type).
    pub(crate) fs: Arc<FilesystemStore>,
    pub(crate) genome: Arc<Genome>,
    pub(crate) coordinate_space: Option<String>,
    pub(crate) tracks: Map<String, Value>,
    pub(crate) track_handles: HashMap<String, Track>,
}

impl PbzStore {
    /// Create a new PBZ store at `path`.
    ///
    /// Writes the root `zarr.json` with `perbase_zarr` metadata, then creates
    /// the `contigs` (string) and `contig_lengths` (int64) 1-D arrays.
    pub fn create(
        path: impl AsRef<Path>,
        genome: Genome,
        coordinate_space: Option<String>,
    ) -> Result<Self> {
        let path = path.as_ref();

        // Open filesystem store at the given path.
        let fs = Arc::new(
            FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?,
        );
        let storage: ReadableWritableListableStorage = fs.clone();

        // Build root group with perbase_zarr attribute.
        let mut root = zarrs::group::GroupBuilder::new()
            .build(storage.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let attrs = root.attributes_mut();

        let root_meta = if let Some(ref cs) = coordinate_space {
            json!({
                "version": PBZ_FORMAT_VERSION,
                "coordinate_space": cs,
                "tracks": {}
            })
        } else {
            json!({
                "version": PBZ_FORMAT_VERSION,
                "tracks": {}
            })
        };
        attrs.insert("perbase_zarr".to_owned(), root_meta);
        root.store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let n = genome.len() as u64;

        // Write the `contigs` 1-D string array.
        let contigs_array = zarrs::array::ArrayBuilder::new(
            vec![n],
            vec![n.max(1)], // chunk = whole array; avoid zero-length chunk
            data_type::string(),
            "",
        )
        .dimension_names(["contigs"].into())
        .build(storage.clone(), "/contigs")
        .map_err(|e| PbzError::Store(e.to_string()))?;
        contigs_array
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        if n > 0 {
            let names: Vec<String> = genome.contigs().iter().map(|c| c.name.clone()).collect();
            let names_array = Array1::from(names).into_dyn();
            contigs_array
                .store_chunk(&[0], names_array)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        // Write the `contig_lengths` 1-D int64 array.
        let lengths_array = zarrs::array::ArrayBuilder::new(
            vec![n],
            vec![n.max(1)],
            data_type::int64(),
            0i64,
        )
        .dimension_names(["contigs"].into())
        .build(storage.clone(), "/contig_lengths")
        .map_err(|e| PbzError::Store(e.to_string()))?;
        lengths_array
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        if n > 0 {
            let lengths: Vec<i64> = genome
                .contigs()
                .iter()
                .map(|c| c.length as i64)
                .collect();
            let lengths_array_nd = Array1::from(lengths).into_dyn();
            lengths_array
                .store_chunk(&[0], lengths_array_nd)
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        Ok(Self {
            storage,
            fs,
            genome: Arc::new(genome),
            coordinate_space,
            tracks: Map::new(),
            track_handles: HashMap::new(),
        })
    }

    /// Open an existing PBZ store at `path` for reading.
    ///
    /// Reads root `perbase_zarr` metadata, then re-hydrates the `Genome` from
    /// the on-disk `contigs` and `contig_lengths` arrays.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Keep the concrete FilesystemStore so Array::open gets a concrete storage
        // type that satisfies ReadableStorageTraits + 'static without trait-object upcasting.
        let fs = Arc::new(
            FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?,
        );
        let storage: ReadableWritableListableStorage = fs.clone();

        // Open root group and extract perbase_zarr attribute.
        let root = zarrs::group::Group::open(fs.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let attrs = root.attributes();
        let pbz_ns = attrs
            .get("perbase_zarr")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                PbzError::Metadata("missing or invalid 'perbase_zarr' root attribute".into())
            })?;

        let coordinate_space = pbz_ns
            .get("coordinate_space")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());

        let tracks: Map<String, Value> = pbz_ns
            .get("tracks")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        // Read the `contigs` 1-D string array.
        let contigs_arr = Array::open(fs.clone(), "/contigs")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let n = contigs_arr.shape()[0] as usize;
        let names: Vec<String> = if n == 0 {
            Vec::new()
        } else {
            let nd: ndarray::ArrayD<String> = contigs_arr
                .retrieve_chunk::<ndarray::ArrayD<String>>(&[0])
                .map_err(|e| PbzError::Store(e.to_string()))?;
            nd.into_raw_vec_and_offset().0
        };

        // Read the `contig_lengths` 1-D int64 array.
        let lengths_arr = Array::open(fs.clone(), "/contig_lengths")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let lengths: Vec<i64> = if n == 0 {
            Vec::new()
        } else {
            let nd: ndarray::ArrayD<i64> = lengths_arr
                .retrieve_chunk::<ndarray::ArrayD<i64>>(&[0])
                .map_err(|e| PbzError::Store(e.to_string()))?;
            nd.into_raw_vec_and_offset().0
        };

        if lengths.len() != names.len() {
            return Err(PbzError::Metadata(format!(
                "contig name/length mismatch: {} names but {} lengths",
                names.len(),
                lengths.len()
            )));
        }

        let contigs: Vec<Contig> = names
            .into_iter()
            .zip(lengths)
            .map(|(name, length)| Contig { name, length: length as u64 })
            .collect();

        let genome = Arc::new(Genome::new(contigs)?);

        // Hydrate track handles from the tracks map.
        let mut track_handles: HashMap<String, Track> = HashMap::new();
        for (name, val) in &tracks {
            let metadata: TrackMetadata = serde_json::from_value(val.clone()).map_err(|e| {
                PbzError::Metadata(format!("invalid track metadata for '{name}': {e}"))
            })?;
            track_handles.insert(
                name.clone(),
                Track {
                    name: name.clone(),
                    metadata,
                    fs: Arc::clone(&fs),
                    genome: Arc::clone(&genome),
                },
            );
        }

        Ok(Self {
            storage,
            fs,
            genome,
            coordinate_space,
            tracks,
            track_handles,
        })
    }

    /// The genome (ordered contig list) for this store.
    pub fn genome(&self) -> &Genome {
        &self.genome
    }

    /// The coordinate space label (e.g. `"GRCh38"`), if set.
    pub fn coordinate_space(&self) -> Option<&str> {
        self.coordinate_space.as_deref()
    }

    /// Iterate over the names of all tracks in this store.
    pub fn track_names(&self) -> impl Iterator<Item = &str> {
        self.tracks.keys().map(|s| s.as_str())
    }

    /// Return a reference to the named track, or `None` if it doesn't exist.
    pub fn track(&self, name: &str) -> Option<&Track> {
        self.track_handles.get(name)
    }

    /// Create a new track in this store.
    ///
    /// Writes a Zarr array under `/<contig>/<track_name>` for every contig in
    /// the genome, then updates the root `perbase_zarr.tracks` attribute.
    ///
    /// Returns an error if the track name already exists.
    pub fn create_track(&mut self, name: &str, config: TrackConfig) -> Result<&Track> {
        if self.tracks.contains_key(name) {
            return Err(PbzError::Metadata(format!(
                "track '{name}' already exists"
            )));
        }

        // Determine the column dimension name for cohort tracks.
        let col_dim: Option<String> = if config.is_cohort() {
            Some(
                config
                    .column_dim
                    .clone()
                    .unwrap_or_else(|| "column".to_owned()),
            )
        } else {
            None
        };
        let columns: Option<&[String]> = config.columns.as_deref();
        let n_cols = columns.map(|c| c.len()).unwrap_or(0) as u64;

        let zarrs_dt = dtype_to_zarrs(config.dtype);
        let default_fill = default_fill_value(config.dtype);

        for contig in self.genome.contigs() {
            let contig_len = contig.length;
            let chunk_pos = (config.chunk_size as u64).min(contig_len).max(1);

            // Ensure the contig group exists.
            let contig_group_path = format!("/{}", contig.name);
            zarrs::group::GroupBuilder::new()
                .build(self.fs.clone(), &contig_group_path)
                .map_err(|e| PbzError::Store(e.to_string()))?
                .store_metadata()
                .map_err(|e| PbzError::Store(e.to_string()))?;

            let data_path = format!("/{}/{}", contig.name, name);

            if let (Some(col_dim_name), Some(cols)) = (col_dim.as_deref(), columns) {
                // 2D cohort array: shape [contig_len, n_cols].
                let col_chunk_size = config
                    .column_chunk_size
                    .map(|s| s as u64)
                    .unwrap_or(n_cols)
                    .min(n_cols)
                    .max(1);

                let arr = zarrs::array::ArrayBuilder::new(
                    vec![contig_len, n_cols],
                    vec![chunk_pos, col_chunk_size],
                    zarrs_dt.clone(),
                    default_fill.clone(),
                )
                .dimension_names(["position", col_dim_name].into())
                .build(self.fs.clone(), &data_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
                arr.store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;

                // Write the coord array: /<contig>/<col_dim> as 1D string array.
                let coord_path = format!("/{}/{}", contig.name, col_dim_name);
                let coord_arr = zarrs::array::ArrayBuilder::new(
                    vec![n_cols],
                    vec![n_cols.max(1)],
                    data_type::string(),
                    "",
                )
                .dimension_names([col_dim_name].into())
                .build(self.fs.clone(), &coord_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
                coord_arr
                    .store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;
                if n_cols > 0 {
                    let labels = Array1::from(cols.to_vec()).into_dyn();
                    coord_arr
                        .store_chunk(&[0], labels)
                        .map_err(|e| PbzError::Store(e.to_string()))?;
                }
            } else {
                // 1D scalar array: shape [contig_len].
                let arr = zarrs::array::ArrayBuilder::new(
                    vec![contig_len],
                    vec![chunk_pos],
                    zarrs_dt.clone(),
                    default_fill.clone(),
                )
                .dimension_names(["position"].into())
                .build(self.fs.clone(), &data_path)
                .map_err(|e| PbzError::Store(e.to_string()))?;
                arr.store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;
            }
        }

        // Build the TrackMetadata and update root attributes.
        let metadata = track_metadata_from_config(&config);
        let meta_val =
            serde_json::to_value(&metadata).map_err(|e| PbzError::Metadata(e.to_string()))?;
        self.tracks.insert(name.to_owned(), meta_val);

        // Rewrite the root zarr.json with updated tracks map.
        let mut root = zarrs::group::Group::open(self.fs.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        let attrs = root.attributes_mut();
        let pbz_ns = attrs
            .get_mut("perbase_zarr")
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| PbzError::Metadata("missing perbase_zarr attribute".into()))?;
        pbz_ns.insert("tracks".to_owned(), Value::Object(self.tracks.clone()));
        root.store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        // Cache the new track handle.
        self.track_handles.insert(
            name.to_owned(),
            Track {
                name: name.to_owned(),
                metadata,
                fs: Arc::clone(&self.fs),
                genome: Arc::clone(&self.genome),
            },
        );

        Ok(self.track_handles.get(name).expect("just inserted"))
    }
}

/// Map `Dtype` → a zarrs `DataType` value.
fn dtype_to_zarrs(d: Dtype) -> zarrs::array::DataType {
    match d {
        Dtype::U8 => data_type::uint8(),
        Dtype::U16 => data_type::uint16(),
        Dtype::U32 => data_type::uint32(),
        Dtype::I8 => data_type::int8(),
        Dtype::I16 => data_type::int16(),
        Dtype::I32 => data_type::int32(),
        Dtype::F32 => data_type::float32(),
        Dtype::F64 => data_type::float64(),
        Dtype::Bool => data_type::bool(),
    }
}

/// Default fill value for a dtype: 0 for ints, NaN for floats, false for bool.
fn default_fill_value(d: Dtype) -> FillValue {
    match d {
        Dtype::U8 => FillValue::from(0u8),
        Dtype::U16 => FillValue::from(0u16),
        Dtype::U32 => FillValue::from(0u32),
        Dtype::I8 => FillValue::from(0i8),
        Dtype::I16 => FillValue::from(0i16),
        Dtype::I32 => FillValue::from(0i32),
        Dtype::F32 => FillValue::from(f32::NAN),
        Dtype::F64 => FillValue::from(f64::NAN),
        Dtype::Bool => FillValue::from(false),
    }
}

/// Convert a `TrackConfig` to the on-disk `TrackMetadata`.
fn track_metadata_from_config(cfg: &TrackConfig) -> TrackMetadata {
    TrackMetadata {
        dtype: cfg.dtype.to_string(),
        chunk_size: cfg.chunk_size,
        column_dim: cfg.column_dim.clone(),
        column_chunk_size: cfg.column_chunk_size,
        shard_size: cfg.shard_size,
        shard_column_size: cfg.shard_column_size,
        fill_value: cfg.fill_value.clone(),
        description: cfg.description.clone(),
        source: cfg.source.clone(),
        extra: cfg.extra.clone(),
    }
}
