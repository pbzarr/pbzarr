//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::{Arc, RwLock};

use hashbrown::HashMap;
use ndarray::Array1;
use serde_json::{Value, json};
use zarrs::array::Array;
use zarrs::array::FillValue;
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ListableStorageTraits, ReadableWritableListableStorage, StorePrefix};

use zarrs::array::codec::api::BytesToBytesCodecTraits;
use zarrs::array::codec::{BloscCodec, BloscCompressionLevel, BloscCompressor, BloscShuffleMode};

use crate::error::PbzError;
use crate::genome::Genome;
use crate::io::Dtype;
use crate::track::{Kind, PerbaseTrackAttrs, Track, TrackConfig, kind_of, rehydrate_genome};
use crate::{PBZ_FORMAT_VERSION, PERBASE_CONVENTION_NAME, PERBASE_CONVENTION_UUID, Result};

/// A handle to an open PBZ store: a Zarr group containing track groups.
pub struct PbzStore {
    pub(crate) storage: ReadableWritableListableStorage,
    pub(crate) track_handles: HashMap<String, Track>,
}

impl PbzStore {
    /// Create a new, empty PBZ store at `path`: a Zarr group whose root
    /// `zarr.json` carries only the `zarr_conventions` marker. Tracks are
    /// added with `create_track`; the store holds no genome of its own.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let fs = FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?;
        let storage: ReadableWritableListableStorage = Arc::new(fs);
        Self::create_with_storage(storage)
    }

    /// Create a new, empty PBZ store over an arbitrary sync zarrs storage
    /// backend (filesystem, in-memory, or a future async-to-sync adapter).
    pub fn create_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        let mut root = zarrs::group::GroupBuilder::new()
            .build(storage.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        root.attributes_mut().insert(
            "zarr_conventions".to_owned(),
            json!([{
                "uuid": PERBASE_CONVENTION_UUID,
                "name": PERBASE_CONVENTION_NAME,
            }]),
        );
        // Node-kind discriminator: this root is a collection (a container of
        // tracks). A track group carries perbase:kind="track" instead.
        root.attributes_mut()
            .insert("perbase:kind".to_owned(), json!("collection"));
        root.attributes_mut()
            .insert("perbase:version".to_owned(), json!(PBZ_FORMAT_VERSION));
        root.store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        Ok(Self {
            storage,
            track_handles: HashMap::new(),
        })
    }

    /// Open an existing PBZ store: enumerate child groups and keep those whose
    /// attributes declare the `perbase` convention, rebuilding each track's
    /// `Genome` from its `contigs` and `offsets` arrays.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let fs = FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?;
        let storage: ReadableWritableListableStorage = Arc::new(fs);
        Self::open_with_storage(storage)
    }

    /// Open an existing PBZ store over an arbitrary sync zarrs storage
    /// backend (filesystem, in-memory, or a future async-to-sync adapter).
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        // Guard: the root must carry the zarr_conventions perbase marker.
        let root = zarrs::group::Group::open(storage.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        if !PerbaseTrackAttrs::conforms(root.attributes()) {
            return Err(PbzError::Metadata(
                "not a pbz 0.4 store (missing zarr_conventions perbase marker); regenerate".into(),
            ));
        }
        if kind_of(root.attributes()) == Kind::Track {
            return Err(PbzError::Metadata(
                "this is a pbz track, not a collection; open it with Track::open".into(),
            ));
        }

        let listing = storage
            .list_dir(&StorePrefix::root())
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let mut track_handles: HashMap<String, Track> = HashMap::new();
        for prefix in listing.prefixes() {
            let name = prefix.as_str().trim_end_matches('/').to_owned();
            if name.is_empty() {
                continue;
            }
            let group = match zarrs::group::Group::open(storage.clone(), &format!("/{name}")) {
                Ok(g) => g,
                Err(_) => continue,
            };
            if !PerbaseTrackAttrs::conforms(group.attributes())
                || kind_of(group.attributes()) != Kind::Track
            {
                // Not a track: either a non-pbz group or a nested collection
                // (reserved, not recursed this round).
                continue;
            }
            let attrs: PerbaseTrackAttrs =
                serde_json::from_value(Value::Object(group.attributes().clone()))
                    .map_err(|e| PbzError::Metadata(format!("track '{name}': {e}")))?;

            let values = Array::open(storage.clone(), &format!("/{name}/values"))
                .map_err(|e| PbzError::Store(e.to_string()))?;
            let dtype = Dtype::from_zarrs(values.data_type())?;
            let rank = values.shape().len();
            let column_dim = if rank == 2 {
                values
                    .dimension_names()
                    .as_ref()
                    .and_then(|d| d.get(1))
                    .and_then(|n| n.clone())
            } else {
                None
            };

            let genome =
                rehydrate_genome(&storage, &format!("/{name}"), attrs.genome_name.as_deref())?;

            track_handles.insert(
                name.clone(),
                Track {
                    name: name.clone(),
                    prefix: format!("/{name}"),
                    genome: Arc::new(genome),
                    dtype,
                    rank,
                    column_dim,
                    storage: Arc::clone(&storage),
                    values: RwLock::new(Some(Arc::new(values))),
                },
            );
        }

        Ok(Self {
            storage,
            track_handles,
        })
    }

    /// Iterate over the names of all tracks in this store.
    pub fn track_names(&self) -> impl Iterator<Item = &str> {
        self.track_handles.keys().map(|s| s.as_str())
    }

    /// Return a reference to the named track, or `None` if it doesn't exist.
    pub fn track(&self, name: &str) -> Option<&Track> {
        self.track_handles.get(name)
    }

    /// The genome of a named track (for resolving regions against it).
    pub fn genome_for(&self, track: &str) -> Option<&Genome> {
        self.track_handles.get(track).map(|t| t.genome.as_ref())
    }

    /// Create a new flat track group `/<name>/` sized to `genome`'s total
    /// length, with `values`, `offsets`, `contigs`, and (for 2D tracks) column-label
    /// arrays. The `perbase:` interpretation block is written to the group
    /// `zarr.json` last, so a reader recognizes the track only once it is
    /// fully created.
    pub fn create_track(
        &mut self,
        name: &str,
        genome: Genome,
        config: TrackConfig,
    ) -> Result<&Track> {
        if self.track_handles.contains_key(name) {
            return Err(PbzError::Metadata(format!("track '{name}' already exists")));
        }

        let total_len: u64 = genome.contigs().iter().map(|c| c.length).sum();
        let zarrs_dt = dtype_to_zarrs(config.dtype);
        let fill = resolve_fill_value(config.dtype, config.fill_value.as_ref())?;
        let data_codecs = default_data_codecs(config.dtype)?;
        let chunk_pos = (config.chunk_size as u64).min(total_len.max(1)).max(1);
        // Shards span full column width, so cross-sample compression still
        // applies within a shard.
        let shard_pos = config
            .shard_size
            .map(|s| (s as u64).min(total_len.max(1)).max(1));

        // The track group.
        zarrs::group::GroupBuilder::new()
            .build(self.storage.clone(), &format!("/{name}"))
            .map_err(|e| PbzError::Store(e.to_string()))?
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let (rank, col_dim): (usize, Option<String>) = match config.columns.as_deref() {
            Some(cols) => {
                let n_cols = cols.len() as u64;
                let col_dim_name = config
                    .column_dim
                    .clone()
                    .unwrap_or_else(|| "column".to_owned());
                let col_chunk = config
                    .column_chunk_size
                    .map(|s| s as u64)
                    .unwrap_or(n_cols)
                    .min(n_cols)
                    .max(1);
                let shard_col = config
                    .shard_column_size
                    .map(|s| (s as u64).min(n_cols).max(1))
                    .unwrap_or(n_cols);
                let outer_chunk = match shard_pos {
                    Some(sp) => vec![sp, shard_col],
                    None => vec![chunk_pos, col_chunk],
                };
                let mut builder = zarrs::array::ArrayBuilder::new(
                    vec![total_len, n_cols],
                    outer_chunk,
                    zarrs_dt.clone(),
                    fill.clone(),
                );
                builder
                    .dimension_names(["position", col_dim_name.as_str()].into())
                    .bytes_to_bytes_codecs(data_codecs.clone());
                if shard_pos.is_some() {
                    builder.subchunk_shape(Some(vec![chunk_pos, col_chunk]));
                }
                builder
                    .build(self.storage.clone(), &format!("/{name}/values"))
                    .map_err(|e| PbzError::Store(e.to_string()))?
                    .store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;

                // Column-label coord array, named after the column dim.
                let coord = zarrs::array::ArrayBuilder::new(
                    vec![n_cols],
                    vec![n_cols.max(1)],
                    data_type::string(),
                    "",
                )
                .dimension_names([col_dim_name.as_str()].into())
                .build(self.storage.clone(), &format!("/{name}/{col_dim_name}"))
                .map_err(|e| PbzError::Store(e.to_string()))?;
                coord
                    .store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;
                if n_cols > 0 {
                    coord
                        .store_chunk(&[0], Array1::from(cols.to_vec()).into_dyn())
                        .map_err(|e| PbzError::Store(e.to_string()))?;
                }
                (2, Some(col_dim_name))
            }
            None => {
                let outer_chunk = match shard_pos {
                    Some(sp) => vec![sp],
                    None => vec![chunk_pos],
                };
                let mut builder = zarrs::array::ArrayBuilder::new(
                    vec![total_len],
                    outer_chunk,
                    zarrs_dt.clone(),
                    fill.clone(),
                );
                builder
                    .dimension_names(["position"].into())
                    .bytes_to_bytes_codecs(data_codecs.clone());
                if shard_pos.is_some() {
                    builder.subchunk_shape(Some(vec![chunk_pos]));
                }
                builder
                    .build(self.storage.clone(), &format!("/{name}/values"))
                    .map_err(|e| PbzError::Store(e.to_string()))?
                    .store_metadata()
                    .map_err(|e| PbzError::Store(e.to_string()))?;
                (1, None)
            }
        };

        // offsets (int64[k+1]) and contigs (vlen-utf8[k]).
        let offsets = genome.offsets();
        let k = genome.len() as u64;
        let off_arr = zarrs::array::ArrayBuilder::new(
            vec![k + 1],
            vec![(k + 1).max(1)],
            data_type::int64(),
            0i64,
        )
        .dimension_names(["contig_boundary"].into())
        .build(self.storage.clone(), &format!("/{name}/offsets"))
        .map_err(|e| PbzError::Store(e.to_string()))?;
        off_arr
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        off_arr
            .store_chunk(&[0], Array1::from(offsets).into_dyn())
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let contigs_arr =
            zarrs::array::ArrayBuilder::new(vec![k], vec![k.max(1)], data_type::string(), "")
                .dimension_names(["contig"].into())
                .build(self.storage.clone(), &format!("/{name}/contigs"))
                .map_err(|e| PbzError::Store(e.to_string()))?;
        contigs_arr
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;
        if k > 0 {
            let names: Vec<String> = genome.contigs().iter().map(|c| c.name.clone()).collect();
            contigs_arr
                .store_chunk(&[0], Array1::from(names).into_dyn())
                .map_err(|e| PbzError::Store(e.to_string()))?;
        }

        // Completion marker: write the perbase block on the group LAST.
        let attrs = PerbaseTrackAttrs::new(&genome, &config);
        let mut group = zarrs::group::Group::open(self.storage.clone(), &format!("/{name}"))
            .map_err(|e| PbzError::Store(e.to_string()))?;
        // Extra attrs first, so the perbase block wins on any key collision.
        for (kk, vv) in &config.extra {
            group.attributes_mut().insert(kk.clone(), vv.clone());
        }
        let attr_val =
            serde_json::to_value(&attrs).map_err(|e| PbzError::Metadata(e.to_string()))?;
        if let Some(obj) = attr_val.as_object() {
            for (kk, vv) in obj {
                group.attributes_mut().insert(kk.clone(), vv.clone());
            }
        }
        group
            .store_metadata()
            .map_err(|e| PbzError::Store(e.to_string()))?;

        let dtype = config.dtype;
        self.track_handles.insert(
            name.to_owned(),
            Track {
                name: name.to_owned(),
                prefix: format!("/{name}"),
                genome: Arc::new(genome),
                dtype,
                rank,
                column_dim: col_dim,
                storage: Arc::clone(&self.storage),
                values: RwLock::new(None),
            },
        );
        Ok(self.track_handles.get(name).expect("just inserted"))
    }
}

/// A polymorphic open result: a pbz artifact on disk is either a single
/// standalone `Track` or a `PbzStore` collection of tracks. `PbzNode::open`
/// reads the root node's `perbase:kind` and reifies the matching one.
pub enum PbzNode {
    Track(Track),
    Collection(PbzStore),
}

impl PbzNode {
    /// Open a pbz artifact at `path`, dispatching on its node kind.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let fs = FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?;
        let storage: ReadableWritableListableStorage = Arc::new(fs);
        Self::open_with_storage(storage)
    }

    /// Open a pbz artifact over an arbitrary storage backend whose root is the
    /// artifact.
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        let root = zarrs::group::Group::open(storage.clone(), "/")
            .map_err(|e| PbzError::Store(e.to_string()))?;
        if !PerbaseTrackAttrs::conforms(root.attributes()) {
            return Err(PbzError::Metadata(
                "not a pbz artifact (missing zarr_conventions perbase marker)".into(),
            ));
        }
        match kind_of(root.attributes()) {
            Kind::Collection => Ok(PbzNode::Collection(PbzStore::open_with_storage(storage)?)),
            Kind::Track => Ok(PbzNode::Track(Track::open_with_storage(storage)?)),
        }
    }
}

/// Default compression: Blosc(zstd, level 5, byte shuffle), per the pbzarr spec.
///
/// Built per-array so `typesize` matches the element size. Returns the
/// vec-form that `ArrayBuilder::bytes_to_bytes_codecs` consumes.
fn default_data_codecs(dtype: Dtype) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>> {
    let typesize = dtype_size(dtype);
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

/// Element size in bytes for each dtype. Drives the blosc `typesize`.
fn dtype_size(d: Dtype) -> usize {
    match d {
        Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
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

/// Resolve a track's fill value: use the caller's `TrackConfig::fill_value`
/// (a JSON scalar) when set, coerced to the track dtype, else the dtype default.
fn resolve_fill_value(d: Dtype, custom: Option<&Value>) -> Result<FillValue> {
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
        Dtype::I8 => {
            FillValue::from(i8::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::I16 => {
            FillValue::from(i16::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
        Dtype::I32 => {
            FillValue::from(i32::try_from(v.as_i64().ok_or_else(&bad)?).map_err(|_| bad())?)
        }
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
        Dtype::I8 => FillValue::from(0i8),
        Dtype::I16 => FillValue::from(0i16),
        Dtype::I32 => FillValue::from(0i32),
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
            Dtype::I8,
            Dtype::I16,
            Dtype::I32,
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
