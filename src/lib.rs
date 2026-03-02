pub mod error;
pub mod region;

pub use error::{PbzError, Result};
pub use region::{parse_region, Region};

/// PBZ specification version implemented by this crate.
pub const PBZ_VERSION: &str = "0.1";
