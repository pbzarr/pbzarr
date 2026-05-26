//! D4-specific ingest glue. `import_d4` body is filled in by Task 11.

use std::path::PathBuf;

use crate::store::PbzStore;
use crate::Result;
use crate::ingest::{ImportConfig, ImportReport};

#[derive(Debug, Clone)]
pub struct D4Source {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Bulk-import one or more d4 files into an existing track.
///
/// Body filled in by Task 11.
pub fn import_d4(
    _store: &mut PbzStore,
    _track_name: &str,
    _sources: &[D4Source],
    _config: ImportConfig,
) -> Result<ImportReport> {
    unimplemented!("import_d4 body — Task 11")
}
