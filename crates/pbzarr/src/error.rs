/// Result alias for PBZ operations.
pub type Result<T> = std::result::Result<T, PbzError>;

/// Errors that can occur when working with PBZ stores.
#[derive(Debug, thiserror::Error)]
pub enum PbzError {
    #[error(
        "This is a format 0.4 store. pbzarr 0.8 reads format 0.5 only. Use pbzarr 0.7, or import the data again."
    )]
    LegacyFormat,
    #[error("Unsupported format version {0}. This pbzarr reads 0.5.")]
    UnsupportedVersion(String),
    #[error("not a valid PBZ store: {0}")]
    InvalidStore(String),
    #[error("unsupported encoding: {0}")]
    UnsupportedEncoding(String),

    #[error("contig not found: {contig} (available: {available:?})")]
    ContigNotFound {
        contig: String,
        available: Vec<String>,
    },

    #[error("invalid region: {message}")]
    InvalidRegion { message: String },

    #[error("invalid dtype: {dtype}")]
    InvalidDtype { dtype: String },

    /// A caller argument that no store or file can satisfy.
    #[error("{0}")]
    InvalidArgument(String),

    #[error("store error: {0}")]
    Store(String),

    #[error("metadata error: {0}")]
    Metadata(String),

    /// The caller set the import cancel flag.
    #[error("import cancelled")]
    Cancelled,

    #[error(transparent)]
    Reader(#[from] crate::io::error::ReaderError),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
