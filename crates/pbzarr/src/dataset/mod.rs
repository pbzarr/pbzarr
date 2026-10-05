//! Reading and source imports for the provisional PBZ root-Dataset convention.
//!
//! These additive APIs do not change format 0.4 or its existing entry points.
mod metadata;
mod read;
mod reference;
mod write;
use crate::{PbzError, Result};
pub use metadata::{
    Role, Support, ValidationIssue, ValidationReport, ValidationStatus, VariableInfo,
};
pub use reference::{BinInterval, GenomicAxis, Reference};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use zarrs::storage::ReadableWritableListableStorage;

/// The `pbz:version` value written to, and required of, a root-Dataset store.
pub const FORMAT_VERSION: &str = "0.5";

/// A metadata catalog of the direct arrays in a root Xarray Dataset.
///
/// Opening reads metadata only. Catalog entries do not guarantee that a
/// variable's encoding is supported or that its values satisfy PBZ invariants.
pub struct Dataset {
    storage: ReadableWritableListableStorage,
    reference_cache: std::sync::OnceLock<Reference>,
    handles: std::sync::Mutex<BTreeMap<String, Arc<read::ArrayHandle>>>,
    root: serde_json::Value,
    variables: BTreeMap<String, VariableInfo>,
}

impl Dataset {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path)
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Self::open_with_storage(Arc::new(storage))
    }
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        let (root, variables) = metadata::catalog(&storage)?;
        Ok(Self {
            storage,
            reference_cache: std::sync::OnceLock::new(),
            handles: std::sync::Mutex::new(BTreeMap::new()),
            root,
            variables,
        })
    }
    pub fn variables(&self) -> &BTreeMap<String, VariableInfo> {
        &self.variables
    }
}
