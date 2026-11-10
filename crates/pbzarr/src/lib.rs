//! Per-base genomic data in a Zarr v3 store, read and written as one root
//! [`Dataset`].
//!
//! [`Dataset`] opens, creates and validates a store. [`Import`] writes
//! per-base data from [`io::ValueReader`] sources into it. [`stat`] reduces a
//! [`Measurement`] over regions.

mod array;
mod codec_spec;
mod consolidated;
mod dataset;
mod error;
pub mod import;
pub mod io;
mod measurement;
mod region_query;
pub mod stat;
mod zip_store;

pub use array::{TrackConfig, validate_node_name};
pub use codec_spec::ExplicitArraySpec;
pub use dataset::{
    BinInterval, Contig, ContigId, Dataset, FORMAT_VERSION, GenomicAxis, Reference, Region, Role,
    StoredSize, Support, ValidationIssue, ValidationReport, ValidationStatus, VariableInfo,
    is_zip_store,
};
pub use error::{PbzError, Result};
pub use import::{Import, ImportBuilder};
pub use measurement::Measurement;
pub use region_query::RegionQuery;

/// The `perbase` convention uuid of a format 0.4 store root, which readers
/// detect to refuse that format.
pub(crate) const PERBASE_CONVENTION_UUID: &str = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c";

/// The `pbz` Zarr convention name (the `zarr_conventions[].name`).
pub const PBZ_CONVENTION_NAME: &str = "pbz";

/// Minted uuid4 for the `pbz` convention. Distinct from the `perbase` uuid so a
/// reader can tell the root-Dataset layout from the per-track group layout.
pub const PBZ_CONVENTION_UUID: &str = "d5bd7bea-0388-4e0a-9844-707c034e9add";
