mod array;
pub mod codec_spec;
mod consolidated;
pub mod dataset;
pub mod error;
pub mod genome;
pub mod import;
pub mod io;
pub mod measurement;
pub mod region_query;
pub mod stat;
mod track;

pub use array::{TrackConfig, validate_column_dimension_name, validate_node_name};
pub use codec_spec::ExplicitArraySpec;
pub use error::{PbzError, Result};
pub use genome::{Contig, ContigId, Genome, Region};
pub use measurement::Measurement;
pub use region_query::{RegionQuery, parse_region_query};
pub use stat::{HistTable, StatKind, StatOptions, StatOutput, StatResult, StatValue};

/// The `perbase` convention uuid of a format 0.4 store root, which readers
/// detect to refuse that format.
pub(crate) const PERBASE_CONVENTION_UUID: &str = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c";

/// The `pbz` Zarr convention name (the `zarr_conventions[].name`).
pub const PBZ_CONVENTION_NAME: &str = "pbz";

/// Minted uuid4 for the `pbz` convention. Distinct from the `perbase` uuid so a
/// reader can tell the root-Dataset layout from the per-track group layout.
pub const PBZ_CONVENTION_UUID: &str = "d5bd7bea-0388-4e0a-9844-707c034e9add";
