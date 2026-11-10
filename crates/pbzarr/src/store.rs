//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::{Arc, RwLock};

use hashbrown::{HashMap, HashSet};
use ndarray::Array1;
use serde_json::{Value, json};
use zarrs::array::Array;
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage, StoreKey,
    StorePrefix, WritableStorageTraits,
};

use crate::array::{
    TrackConfig, apply_explicit_spec, default_data_codecs, dtype_to_zarrs, resolve_fill_value,
};
pub use crate::array::{validate_column_dimension_name, validate_node_name};
use crate::error::PbzError;
use crate::genome::{Contig, Genome};
use crate::io::Dtype;
use crate::track::{PendingTrack, PerbaseTrackAttrs, Track};
use crate::{PERBASE_CONVENTION_NAME, PERBASE_CONVENTION_UUID, Result};

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
            if !PerbaseTrackAttrs::conforms(group.attributes()) {
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

            let genome = rehydrate_genome(&storage, &name, attrs.genome_name.as_deref())?;

            track_handles.insert(
                name.clone(),
                Track {
                    name: name.clone(),
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
    /// length, with `values`, `offsets`, `contigs`, and (cohort) column-label
    /// arrays. The `perbase:` interpretation block is written to the group
    /// `zarr.json` last, so a reader recognizes the track only once it is
    /// fully created.
    pub fn create_track(
        &mut self,
        name: &str,
        genome: Genome,
        config: TrackConfig,
    ) -> Result<&Track> {
        let name = name.to_owned();
        self.create_tracks_with(vec![(name.clone(), genome, config)], |_| Ok(()))?;
        self.track_handles
            .get(&name)
            .ok_or_else(|| PbzError::Store(format!("track {name:?} missing after publication")))
    }

    /// Create one or more unpublished tracks, populate them through `populate`,
    /// and publish each track only after the closure succeeds.
    pub fn create_tracks_with<T>(
        &mut self,
        specs: Vec<(String, Genome, TrackConfig)>,
        populate: impl FnOnce(&[&Track]) -> Result<T>,
    ) -> Result<T> {
        if specs.is_empty() {
            return Err(PbzError::Metadata("no tracks to create".into()));
        }

        let mut names = HashSet::with_capacity(specs.len());
        for (name, _, config) in &specs {
            validate_node_name(name)?;
            if config.columns.is_some() {
                validate_column_dimension_name(config.column_dim.as_deref().unwrap_or("column"))?;
            }
            if !names.insert(name.clone()) || self.track_handles.contains_key(name) {
                return Err(PbzError::Metadata(format!("track '{name}' already exists")));
            }
            let group_key = StoreKey::new(format!("{name}/zarr.json"))
                .map_err(|e| PbzError::Metadata(e.to_string()))?;
            if self
                .storage
                .get(&group_key)
                .map_err(|e| PbzError::Store(e.to_string()))?
                .is_some()
            {
                return Err(PbzError::Metadata(format!(
                    "track '{name}' has an existing physical group"
                )));
            }
        }

        let mut pending = Vec::with_capacity(specs.len());
        for (name, genome, config) in specs {
            pending.push(self.create_pending_track(&name, genome, config)?);
        }

        let tracks: Vec<&Track> = pending.iter().map(|pending| &pending.track).collect();
        let result = populate(&tracks)?;
        drop(tracks);

        for pending_track in pending {
            let name = pending_track.track.name.clone();
            let track = pending_track.finalize()?;
            self.track_handles.insert(name, track);
        }

        // Track completion is a publication event: refresh the consolidated
        // metadata last, once for the whole batch, so a reader resolving
        // solely from the root map sees the new tracks.
        self.consolidate_metadata()?;

        Ok(result)
    }

    fn create_pending_track(
        &self,
        name: &str,
        genome: Genome,
        config: TrackConfig,
    ) -> Result<PendingTrack> {
        let total_len: u64 = genome.contigs().iter().map(|c| c.length).sum();
        let zarrs_dt = dtype_to_zarrs(config.dtype);
        let fill = resolve_fill_value(config.dtype, config.fill_value.as_ref())?;
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
                let outer_chunk = if config.codecs.is_some() {
                    vec![chunk_pos, col_chunk]
                } else {
                    match shard_pos {
                        Some(sp) => vec![sp, shard_col],
                        None => vec![chunk_pos, col_chunk],
                    }
                };
                let mut builder = zarrs::array::ArrayBuilder::new(
                    vec![total_len, n_cols],
                    outer_chunk,
                    zarrs_dt.clone(),
                    fill.clone(),
                );
                builder.dimension_names(["position", col_dim_name.as_str()].into());
                if let Some(spec) = &config.codecs {
                    apply_explicit_spec(&mut builder, name, spec, &[total_len, n_cols])?;
                } else {
                    builder.bytes_to_bytes_codecs(default_data_codecs(config.dtype)?);
                    if shard_pos.is_some() {
                        builder.subchunk_shape(Some(vec![chunk_pos, col_chunk]));
                    }
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
                let outer_chunk = if config.codecs.is_some() {
                    vec![chunk_pos]
                } else {
                    match shard_pos {
                        Some(sp) => vec![sp],
                        None => vec![chunk_pos],
                    }
                };
                let mut builder = zarrs::array::ArrayBuilder::new(
                    vec![total_len],
                    outer_chunk,
                    zarrs_dt.clone(),
                    fill.clone(),
                );
                builder.dimension_names(["position"].into());
                if let Some(spec) = &config.codecs {
                    apply_explicit_spec(&mut builder, name, spec, &[total_len])?;
                } else {
                    builder.bytes_to_bytes_codecs(default_data_codecs(config.dtype)?);
                    if shard_pos.is_some() {
                        builder.subchunk_shape(Some(vec![chunk_pos]));
                    }
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

        let dtype = config.dtype;
        let track = Track {
            name: name.to_owned(),
            genome: Arc::new(genome),
            dtype,
            rank,
            column_dim: col_dim,
            storage: Arc::clone(&self.storage),
            values: RwLock::new(None),
        };
        PendingTrack::new(track, &config)
    }

    /// Refresh the store-root consolidated metadata (zarr-python v3 flavor).
    ///
    /// The consolidated copy is advisory: the per-node `zarr.json` documents
    /// stay authoritative and are not modified.
    pub fn consolidate_metadata(&self) -> Result<()> {
        let root_key = StoreKey::new("zarr.json").expect("root key");
        let bytes = self
            .storage
            .get(&root_key)
            .map_err(|e| PbzError::Store(e.to_string()))?
            .ok_or_else(|| PbzError::Metadata("store root has no zarr.json".into()))?;
        let mut root: Value = serde_json::from_slice(&bytes)
            .map_err(|e| PbzError::Metadata(format!("store root: invalid zarr.json: {e}")))?;
        crate::consolidated::insert_snapshot(&self.storage, &mut root)?;
        let encoded = serde_json::to_vec_pretty(&root)
            .map_err(|e| PbzError::Metadata(format!("store root: encode zarr.json: {e}")))?;
        self.storage
            .set(&root_key, encoded.into())
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Ok(())
    }
}

/// Rebuild a track's `Genome` from its on-disk `contigs` and `offsets` arrays.
fn rehydrate_genome(
    storage: &ReadableWritableListableStorage,
    track_name: &str,
    genome_name: Option<&str>,
) -> Result<Genome> {
    let contigs_arr = Array::open(storage.clone(), &format!("/{track_name}/contigs"))
        .map_err(|e| PbzError::Store(e.to_string()))?;
    let k = contigs_arr.shape()[0] as usize;
    let names: Vec<String> = if k == 0 {
        Vec::new()
    } else {
        contigs_arr
            .retrieve_chunk::<ndarray::ArrayD<String>>(&[0])
            .map_err(|e| PbzError::Store(e.to_string()))?
            .into_raw_vec_and_offset()
            .0
    };
    let offsets_arr = Array::open(storage.clone(), &format!("/{track_name}/offsets"))
        .map_err(|e| PbzError::Store(e.to_string()))?;
    let offsets: Vec<i64> = offsets_arr
        .retrieve_chunk::<ndarray::ArrayD<i64>>(&[0])
        .map_err(|e| PbzError::Store(e.to_string()))?
        .into_raw_vec_and_offset()
        .0;
    if offsets.len() != k + 1 {
        return Err(PbzError::Metadata(format!(
            "track '{track_name}': offsets len {} != contigs {} + 1",
            offsets.len(),
            k
        )));
    }
    let contigs: Vec<Contig> = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| Contig {
            name,
            length: (offsets[i + 1] - offsets[i]) as u64,
        })
        .collect();
    let mut genome = Genome::new(contigs)?;
    if let Some(n) = genome_name {
        genome = genome.with_name(n);
    }
    Ok(genome)
}
