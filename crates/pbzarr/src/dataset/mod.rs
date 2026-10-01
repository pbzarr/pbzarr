//! Reader for the provisional PBZ root-Dataset convention.
//!
//! This module does not change the existing format 0.4 reader or writer.
mod reference;
pub use reference::{BinInterval, GenomicAxis, Reference};
/// Unpublished encoding identifier, independent of the package version.
pub const FORMAT_VERSION: &str = "core-draft-1";
