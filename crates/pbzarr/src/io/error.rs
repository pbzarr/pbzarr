use std::path::PathBuf;

use crate::io::dtype::Dtype;

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

    #[error("malformed record in {path} at line {line}: {message}", path = path.display())]
    Parse {
        path: PathBuf,
        line: u64,
        message: String,
    },

    #[error("contig {contig:?} not found in {path}", path = path.display())]
    ContigNotFound { path: PathBuf, contig: String },

    #[error(
        "overlapping intervals in {path} at {contig}:{this_start}-{this_end} \
         (previous ended at {prev_end})",
        path = path.display()
    )]
    Overlap {
        path: PathBuf,
        contig: String,
        this_start: u64,
        this_end: u64,
        prev_end: u64,
    },

    #[error("dtype mismatch in {path}: expected {expected}, source has {found}", path = path.display())]
    DtypeMismatch {
        path: PathBuf,
        expected: Dtype,
        found: Dtype,
    },

    #[error("value out of range for {dtype} at {path}:{line}: {value}", path = path.display())]
    ValueOutOfRange {
        path: PathBuf,
        line: u64,
        dtype: Dtype,
        value: String,
    },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result alias for reader operations.
pub type Result<T> = std::result::Result<T, ReaderError>;
