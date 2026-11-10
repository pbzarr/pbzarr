//! Reading and source imports for the PBZ root-Dataset convention.
mod metadata;
mod read;
mod reference;
mod write;
use crate::{PbzError, Result};
pub use metadata::{
    Role, Support, ValidationIssue, ValidationReport, ValidationStatus, VariableInfo,
};
pub(crate) use read::float_fill;
pub use reference::{BinInterval, Contig, ContigId, GenomicAxis, Reference, Region};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};
pub(crate) use write::DatasetPlan;
use zarrs::storage::ReadableWritableListableStorage;

/// The `pbz:version` value written to, and required of, a root-Dataset store.
pub const FORMAT_VERSION: &str = "0.5";

/// The dimension of the reference arrays.
pub(crate) const CONTIG_DIM: &str = "contig";
pub(crate) const CONTIG_ID: &str = "contig_id";
pub(crate) const CONTIG_LENGTH: &str = "contig_length";
pub(crate) const CONTIG_MD5: &str = "contig_md5";
/// Root arrays that are coordinates by name. Offsets come from a prefix sum
/// of `contig_length`.
pub(crate) const REFERENCE_ARRAYS: [&str; 3] = [CONTIG_ID, CONTIG_LENGTH, CONTIG_MD5];

/// A metadata catalog of the direct arrays in a root Xarray Dataset.
///
/// Opening reads metadata only. Catalog entries do not guarantee that a
/// variable's encoding is supported or that its values satisfy PBZ invariants.
pub struct Dataset {
    storage: ReadableWritableListableStorage,
    /// The store directory when the storage is a filesystem store.
    dir: Option<PathBuf>,
    reference_cache: std::sync::OnceLock<Reference>,
    handles: std::sync::Mutex<BTreeMap<String, Arc<read::ArrayHandle>>>,
    root: serde_json::Value,
    variables: BTreeMap<String, VariableInfo>,
}

impl Dataset {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path.as_ref())
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Self::open_in(Arc::new(storage), Some(path.as_ref().to_path_buf()))
    }
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        Self::open_in(storage, None)
    }
    fn open_in(storage: ReadableWritableListableStorage, dir: Option<PathBuf>) -> Result<Self> {
        let (root, variables) = metadata::catalog(&storage)?;
        Ok(Self {
            storage,
            dir,
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
