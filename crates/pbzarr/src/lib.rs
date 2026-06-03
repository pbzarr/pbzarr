pub mod error;
pub mod genome;
pub mod import;
pub mod io;
pub mod region_query;
pub mod store;
pub mod track;

pub use error::{PbzError, Result};
pub use genome::{Contig, ContigId, Genome, Region};
pub use region_query::{RegionQuery, parse_region_query};
pub use store::PbzStore;
pub use track::{Track, TrackConfig, TrackMetadata};

/// PBZ on-disk format version this implementation writes.
pub const PBZ_FORMAT_VERSION: &str = "0.1";
