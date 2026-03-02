pub mod error;
pub mod region;
pub mod store;

pub use error::{PbzError, Result};
pub use region::{parse_region, Region};
pub use store::PbzStore;

/// PBZ specification version implemented by this crate.
pub const PERBASE_ZARR_VERSION: &str = "0.1";
