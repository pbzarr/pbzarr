use std::path::PathBuf;

/// Errors returned by `ValueReader` impls.
///
/// Every variant carries the source path so cohort-scale import can attribute
/// failures to a specific input. Orchestrators decorate further (which sample,
/// which region) at their layer.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("I/O error in {path}: {source}", path = path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("contig {contig:?} not found in {path}", path = path.display())]
    ContigNotFound { path: PathBuf, contig: String },

    #[error("schema mismatch: {message}")]
    SchemaMismatch { message: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result alias for reader operations.
pub type Result<T> = std::result::Result<T, ReaderError>;
