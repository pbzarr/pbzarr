pub mod error;
pub mod region_query;
pub mod store;
pub mod track;
pub mod ingest;
pub mod io;
pub mod genome;

pub use error::{PbzError, Result};
pub use region_query::{RegionQuery, parse_region_query};
pub use genome::{Contig, ContigId, Genome, Region};

/// PBZ on-disk format version this implementation writes.
pub const PBZ_FORMAT_VERSION: &str = "0.1";
