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
    BedColumnSpec, BedImportOptions, BedLayout, BedMultiReader, BedSchema, BedSchemaPlan,
    ColumnSelector, InferRows, column_index_by_name, execute_bed_schema_plan, from_bed_matrix,
    infer_bed_dtypes, infer_bed_dtypes_for_sources, plan_bed_schema, read_bed_layout,
};
pub use bigwig::{BigWigReader, from_bigwig};
pub use d4::{D4Reader, from_d4};
pub use pbzarr::import::Source;
