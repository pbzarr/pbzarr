//! Input-format readers for pbzarr. This crate owns format-specific git
//! dependencies (currently `d4`), keeping them out of the core `pbzarr`
//! crate's manifest so `pbzarr` stays publishable to crates.io (which rejects
//! git dependencies). Each format's reader plugs into the core `ValueReader`
//! pipeline; the `from_<format>` functions are the bulk-import entry points.
//! Future formats (bed, bedgraph) will land as sibling modules.

pub mod bam;
pub mod bed;
pub mod bigwig;
pub(crate) mod coords;
pub mod d4;

pub use bam::{BamReader, DepthFilter, ImportMode, OverlapMode, from_bam};
pub use bed::{
    BedColumnSpec, BedImportOptions, BedLayout, BedMultiReader, BedReader, BedSchema,
    ColumnSelector, InferRows, column_index_by_name, from_bed, from_bed_matrix, from_bed_multi,
    infer_bed_dtypes, read_bed_layout,
};
pub use bigwig::{BigWigReader, from_bigwig};
pub use d4::{D4Reader, from_d4};
pub use pbzarr::import::Source;
