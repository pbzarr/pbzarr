//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::Arc;

use ndarray::Array1;
use serde_json::{Map, Value, json};
use zarrs::array::Array;
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::PbzError;
use crate::genome::{Contig, Genome};
use crate::{PBZ_FORMAT_VERSION, Result};

/// A handle to an open PBZ store.
pub struct PbzStore {
    // Storage handle; used by write operations in later tasks.
    #[allow(dead_code)]
    pub(crate) storage: ReadableWritableListableStorage,
    pub(crate) genome: Genome,
    pub(crate) coordinate_space: Option<String>,
    pub(crate) tracks: Map<String, Value>,
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
        let storage: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path).map_err(|e| PbzError::Store(e.to_string()))?);

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
            genome,
            coordinate_space,
            tracks: Map::new(),
        })
    }

    /// Open an existing PBZ store at `path` for reading.
    ///
    /// Reads root `perbase_zarr` metadata, then re-hydrates the `Genome` from
    /// the on-disk `contigs` and `contig_lengths` arrays.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // Use FilesystemStore directly so Array::open gets a concrete storage type
        // that satisfies ReadableStorageTraits + 'static without trait-object upcasting.
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

        let genome = Genome::new(contigs)?;

        Ok(Self {
            storage,
            genome,
            coordinate_space,
            tracks,
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
}
