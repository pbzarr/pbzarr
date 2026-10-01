//! Reader for the provisional PBZ root-Dataset convention.
//!
//! This additive reader does not change format 0.4 or its existing opener.
mod metadata;
mod reference;
use crate::{PbzError, Result};
pub use metadata::{Role, Support, VariableInfo};
pub use reference::{BinInterval, GenomicAxis, Reference};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use zarrs::storage::ReadableWritableListableStorage;

/// Development convention token, not a published format version.
pub const FORMAT_VERSION: &str = "core-draft-1";

/// A metadata catalog of the direct arrays in a root Xarray Dataset.
///
/// Opening reads metadata only. Catalog entries do not guarantee that a
/// variable's encoding is supported or that its values satisfy PBZ invariants.
pub struct Dataset {
    storage: ReadableWritableListableStorage,
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
            root,
            variables,
        })
    }
    pub fn variables(&self) -> &BTreeMap<String, VariableInfo> {
        &self.variables
    }
}
