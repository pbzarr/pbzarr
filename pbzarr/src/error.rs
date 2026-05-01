/// Result alias for PBZ operations.
pub type Result<T> = std::result::Result<T, PbzError>;

/// Errors that can occur when working with PBZ stores.
#[derive(Debug, thiserror::Error)]
pub enum PbzError {
    #[error("contig not found: {contig} (available: {available:?})")]
    ContigNotFound {
        contig: String,
        available: Vec<String>,
    },

    #[error("track not found: {name} (available: {available:?})")]
    TrackNotFound {
        name: String,
        available: Vec<String>,
    },

    #[error("column not found: {name} (available: {available:?})")]
    ColumnNotFound {
        name: String,
        available: Option<Vec<String>>,
    },

    #[error("invalid region: {message}")]
    InvalidRegion { message: String },

    #[error("invalid dtype: {dtype}")]
    InvalidDtype { dtype: String },

    #[error("store error: {0}")]
    Store(String),

    #[error("metadata error: {0}")]
    Metadata(String),

    #[error("track {track:?} missing required 'layout' attribute")]
    MissingLayout { track: String },

    #[error("track {track:?} has unrecognized layout {layout:?}")]
    UnknownLayout { track: String, layout: String },

    #[error(
        "single-sample data MUST be stored as a 1D track (no samples array); \
         use TrackConfig without samples for a single sample"
    )]
    SingleSampleMustBe1D,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
