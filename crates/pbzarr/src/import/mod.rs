//! Import pipeline. Bulk-write per-base data from `ValueReader` sources
//! into a `PbzStore` track via a `crossbeam-channel` worker pool.
//!
//! The pipeline is format-agnostic: it drives any `ValueReader`. Format
//! bindings (e.g. d4) live in their own crates and call `run_pipeline`.

mod pipeline;
pub mod progress;
mod source;

pub use pipeline::{
    Config, ProgressSink, Report, run_matrix_pipeline, run_multi_pipeline, run_pipeline,
    run_wide_pipeline,
};
pub use source::Source;
