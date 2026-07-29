pub mod error;
pub mod genome;
pub mod import;
pub mod io;
pub mod region_query;
pub mod region_store;
pub mod stack;
pub mod store;
pub mod track;

pub use error::{PbzError, Result};
pub use genome::{Contig, ContigId, Genome, Region};
pub use region_query::{RegionQuery, parse_region_query};
pub use region_store::{RegionBuildConfig, build_region_store};
pub use stack::{ProgressFactory, StackConfig, stack};
pub use store::{PbzNode, PbzStore, Segmentation};
pub use track::{ConventionRef, Kind, PerbaseTrackAttrs, Track, TrackConfig, kind_of};

/// The pbz format/convention version written to every track group.
pub const PBZ_FORMAT_VERSION: &str = "0.4";

/// The `perbase` Zarr convention name (the `zarr_conventions[].name`).
pub const PERBASE_CONVENTION_NAME: &str = "perbase";

/// Minted uuid4 for the `perbase` convention. Stable across releases; the
/// registry entry / schema_url are published later (design-to-conform-now).
pub const PERBASE_CONVENTION_UUID: &str = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c";
