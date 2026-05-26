//! `PbzStore`: top-level handle for a PBZ on-disk store.

use std::path::Path;
use std::sync::Arc;

use ndarray::Array1;
use serde_json::{Map, Value, json};
use zarrs::array::data_type;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::PbzError;
use crate::genome::Genome;
use crate::{PBZ_FORMAT_VERSION, Result};

/// A handle to an open PBZ store.
pub struct PbzStore {
    #[allow(dead_code)]
    pub(crate) storage: ReadableWritableListableStorage,
    #[allow(dead_code)]
    pub(crate) genome: Genome,
    #[allow(dead_code)]
    pub(crate) coordinate_space: Option<String>,
    #[allow(dead_code)]
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
}
