//! Ingest pipeline. Bulk-write per-base data from `ValueReader` sources
//! into a `PbzStore` track via a `crossbeam-channel` worker pool.
//!
//! Format-specific submodules (d4, future bigWig/BED) wire their reader
//! into the generic pipeline.

mod pipeline;
mod d4_import;

pub use pipeline::{ImportConfig, ImportReport, ProgressSink, run_pipeline};
pub use d4_import::{D4Source, import_d4};
