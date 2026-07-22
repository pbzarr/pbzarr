//! Input-format readers for pbzarr. This crate owns format-specific git
//! dependencies (currently `d4`), keeping them out of the core `pbzarr`
//! crate's manifest so `pbzarr` stays publishable to crates.io (which rejects
//! git dependencies). Each format's reader plugs into the core `ValueReader`
//! pipeline; the `from_<format>` functions are the bulk-import entry points.
//! Future formats (bed, bedgraph) will land as sibling modules.

pub mod bed;
pub mod bigwig;
pub mod d4;

pub use bed::{
    BedColumnSpec, BedMultiReader, BedReader, BedSchema, BedSource, ColumnSelector,
    column_index_by_name, from_bed, from_bed_multi,
};
pub use bigwig::{BigWigReader, BigWigSource, from_bigwig};
pub use d4::{D4Reader, D4Source, from_d4};
