pub mod error;
pub mod genome;
pub mod import;
pub mod io;
pub mod region_query;
pub mod stack;
pub mod store;
pub mod track;

pub use error::{PbzError, Result};
pub use genome::{Contig, ContigId, Genome, Region};
pub use region_query::{RegionQuery, parse_region_query};
#[allow(deprecated)]
pub use stack::{StackConfig, stack};
pub use store::{PbzStore, validate_column_dimension_name, validate_node_name};
pub use track::{ConventionRef, PerbaseTrackAttrs, Track, TrackConfig};

/// The pbz format/convention version written to every track group.
pub const PBZ_FORMAT_VERSION: &str = "0.4";

/// The `perbase` Zarr convention name (the `zarr_conventions[].name`).
pub const PERBASE_CONVENTION_NAME: &str = "perbase";

/// Minted uuid4 for the `perbase` convention. Stable across releases; the
/// registry entry / schema_url are published later (design-to-conform-now).
pub const PERBASE_CONVENTION_UUID: &str = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c";
