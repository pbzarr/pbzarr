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
    BedColumnRequest, BedColumnSpec, BedImportOptions, BedLayout, BedMultiReader, BedSchema,
    BedSchemaPlan, ColumnSelector, InferRows, build_bed_schema, column_index_by_name,
    execute_bed_schema_plan, from_bed_matrix, infer_bed_dtypes, infer_bed_dtypes_for_sources,
    plan_bed_schema, read_bed_layout,
};
pub use bigwig::{BigWigReader, from_bigwig};
pub use d4::{D4Reader, from_d4};
pub use pbzarr::import::Source;

use std::path::Path;

use pbzarr::io::{ReaderError, ValueReader};
use pbzarr::{PbzError, Result};

/// A source that did not open. A file system error keeps its kind, so a
/// caller can tell a missing file from a bad one.
pub(crate) fn open_failed(path: &Path, error: ReaderError) -> PbzError {
    match error {
        ReaderError::Io { .. } => PbzError::Reader(error),
        other => PbzError::Store(format!("open {}: {other}", path.display())),
    }
}

/// Every source must list the same contig names and lengths in the same
/// order as the first source.
pub(crate) fn check_same_contigs<R: ValueReader>(
    format: &str,
    readers: &[R],
    sources: &[Source],
) -> Result<()> {
    let Some(first) = readers.first() else {
        return Ok(());
    };
    for (reader, source) in readers.iter().zip(sources).skip(1) {
        if reader.contigs().contigs() != first.contigs().contigs() {
            return Err(PbzError::Metadata(format!(
                "{format} import: the contigs of {} differ from {} (names and lengths must match in order)",
                source.path.display(),
                sources[0].path.display()
            )));
        }
    }
    Ok(())
}
