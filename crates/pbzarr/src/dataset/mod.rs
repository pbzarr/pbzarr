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
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorage};

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
    reader: ReadableStorage,
    /// `None` when the Dataset is open read-only.
    writer: Option<ReadableWritableListableStorage>,
    /// The store directory when the storage is a filesystem store.
    dir: Option<PathBuf>,
    reference_cache: std::sync::OnceLock<Reference>,
    handles: std::sync::Mutex<BTreeMap<String, Arc<read::ArrayHandle>>>,
    root: serde_json::Value,
    variables: BTreeMap<String, VariableInfo>,
}

impl Dataset {
    /// Open a directory store for reads and writes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let storage = zarrs::filesystem::FilesystemStore::new(path.as_ref())
            .map_err(|e| PbzError::Store(e.to_string()))?;
        Self::open_in(Arc::new(storage), Some(path.as_ref().to_path_buf()))
    }
    pub fn open_with_storage(storage: ReadableWritableListableStorage) -> Result<Self> {
        Self::open_in(storage, None)
    }
    /// Open a directory store for reads only. The store is listed to find
    /// its variables, as [`Dataset::open`] does.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let storage = Arc::new(
            zarrs::filesystem::FilesystemStore::new(path.as_ref())
                .map_err(|e| PbzError::Store(e.to_string()))?,
        );
        Self::open_reader(
            storage.clone(),
            metadata::Nodes::Listed(storage),
            Some(path.as_ref().to_path_buf()),
        )
    }
    /// Open a store for reads only over storage that need not list or write.
    ///
    /// The variable list comes from the root consolidated metadata, so the
    /// store must have it.
    pub fn open_readable(storage: ReadableStorage) -> Result<Self> {
        Self::open_reader(storage, metadata::Nodes::Consolidated, None)
    }
    fn open_reader(
        reader: ReadableStorage,
        nodes: metadata::Nodes,
        dir: Option<PathBuf>,
    ) -> Result<Self> {
        let (root, variables) = metadata::catalog(&reader, nodes)?;
        Ok(Self::new(reader, None, dir, root, variables))
    }
    fn open_in(storage: ReadableWritableListableStorage, dir: Option<PathBuf>) -> Result<Self> {
        let reader: ReadableStorage = storage.clone();
        let (root, variables) =
            metadata::catalog(&reader, metadata::Nodes::Listed(storage.clone()))?;
        Ok(Self::new(reader, Some(storage), dir, root, variables))
    }
    fn new(
        reader: ReadableStorage,
        writer: Option<ReadableWritableListableStorage>,
        dir: Option<PathBuf>,
        root: serde_json::Value,
        variables: BTreeMap<String, VariableInfo>,
    ) -> Self {
        Self {
            reader,
            writer,
            dir,
            reference_cache: std::sync::OnceLock::new(),
            handles: std::sync::Mutex::new(BTreeMap::new()),
            root,
            variables,
        }
    }
    /// The write handle, or an error when the Dataset is open read-only.
    pub(crate) fn writable(&self) -> Result<&ReadableWritableListableStorage> {
        self.writer.as_ref().ok_or_else(|| {
            PbzError::Store(
                "this store is open read-only. Open it with Dataset::open to import or append"
                    .into(),
            )
        })
    }
    pub fn variables(&self) -> &BTreeMap<String, VariableInfo> {
        &self.variables
    }
}
