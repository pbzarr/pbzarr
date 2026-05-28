//! Import pipeline. Bulk-write per-base data from `ValueReader` sources
//! into a `PbzStore` track via a `crossbeam-channel` worker pool.
//!
//! Format-specific submodules (d4, future bigWig/BED) wire their reader
//! into the generic pipeline.

mod d4;
mod pipeline;

pub use d4::{D4Source, from_d4};
pub use pipeline::{Config, ProgressSink, Report, run_pipeline};
