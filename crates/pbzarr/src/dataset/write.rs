//! Minimal creation and import support for the root Dataset convention.
use super::{Dataset, FORMAT_VERSION, Reference};
use crate::{PbzError, Result};
use std::{path::Path, sync::Arc};
use zarrs::{
    array::{ArrayBuilder, data_type},
    storage::{ReadableWritableListableStorage, StorePrefix},
};

fn storage_error(error: impl std::fmt::Display) -> PbzError {
    PbzError::Store(error.to_string())
}

impl Dataset {
    /// Create a new core-draft-1 Dataset with a fixed ordered reference.
    ///
    /// The storage namespace must be empty. No conversion or replacement is performed.
    pub fn create(path: impl AsRef<Path>, reference: Reference) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path).map_err(storage_error)?;
        Self::create_with_storage(Arc::new(storage), reference)
    }

    /// Create a Dataset in an empty storage namespace.
    ///
    /// Reference arrays are written before root metadata. Errors may leave
    /// incomplete output; automatic cleanup and recovery are not implemented.
    pub fn create_with_storage(
        storage: ReadableWritableListableStorage,
        reference: Reference,
    ) -> Result<Self> {
        let entries = storage
            .list_dir(&StorePrefix::root())
            .map_err(storage_error)?;
        if !entries.keys().is_empty() || !entries.prefixes().is_empty() {
            return Err(PbzError::Metadata(
                "Dataset creation requires an empty storage namespace".into(),
            ));
        }
        let n = reference.contigs().len() as u64;
        let names = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::string(), "")
            .dimension_names(["contig"].into())
            .build(storage.clone(), "/contig")
            .map_err(storage_error)?;
        let lengths = ArrayBuilder::new(vec![n], vec![n.max(1)], data_type::int64(), 0_i64)
            .dimension_names(["contig"].into())
            .build(storage.clone(), "/contig_length")
            .map_err(storage_error)?;
        if n > 0 {
            let values: Vec<String> = reference.contigs().iter().map(|c| c.name.clone()).collect();
            names
                .store_chunk(&[0], ndarray::Array1::from(values))
                .map_err(storage_error)?;
            let values: Vec<i64> = reference
                .contigs()
                .iter()
                .map(|c| c.length as i64)
                .collect();
            lengths
                .store_chunk(&[0], ndarray::Array1::from(values))
                .map_err(storage_error)?;
        }
        names.store_metadata().map_err(storage_error)?;
        lengths.store_metadata().map_err(storage_error)?;
        let mut root = zarrs::group::GroupBuilder::new()
            .build(storage.clone(), "/")
            .map_err(storage_error)?;
        root.attributes_mut().extend(
            serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(
                serde_json::json!({
                    "zarr_conventions": [{"uuid":crate::PERBASE_CONVENTION_UUID,"name":"pbz:"}],
                    "pbz:version":FORMAT_VERSION,
                    "pbz:reference":{"names":"contig","lengths":"contig_length"},
                    "coordinates":"contig_length"
                }),
            )
            .expect("literal attribute object"),
        );
        root.store_metadata().map_err(storage_error)?;
        Self::open_with_storage(storage)
    }
}
