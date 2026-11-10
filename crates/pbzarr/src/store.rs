//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::{Arc, RwLock};

use hashbrown::HashMap;
use serde_json::Value;
use zarrs::array::Array;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::{ListableStorageTraits, ReadableWritableListableStorage, StorePrefix};

use crate::Result;
pub use crate::array::{validate_column_dimension_name, validate_node_name};
use crate::error::PbzError;
use crate::genome::{Contig, Genome};
use crate::io::Dtype;
use crate::track::{PerbaseTrackAttrs, Track};

/// A handle to an open PBZ store: a Zarr group containing track groups.
pub struct PbzStore {
    pub(crate) track_handles: HashMap<String, Track>,
}

impl PbzStore {
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

        Ok(Self { track_handles })
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
