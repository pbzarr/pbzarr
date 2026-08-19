//! Import: bulk-write per-base data from `ValueReader` sources into `PbzStore`
//! tracks via a `crossbeam-channel` worker pool.
//!
//! The engine is format-agnostic: it drives any `ValueReader`. Format readers
//! (d4, bigWig, BED, BAM/CRAM) live in their own crates and start runs through
//! [`Import`].

mod builder;
mod config;
mod engine;
mod estimate;
pub mod progress;
mod routing;
mod source;

pub use builder::{Import, ImportBuilder};
pub use config::{Config, ProgressSink, Report};
pub use engine::{CostModel, PipelineOptions, TapMessage, auto_in_flight_spans};
pub use estimate::{
    GenomeGeometry, LayoutEstimate, LayoutKnobs, LevelEstimate, TrackEstimate, TrackShape,
};
pub use routing::{ImportRouting, SourceAxis, TrackTarget};
pub use source::Source;
