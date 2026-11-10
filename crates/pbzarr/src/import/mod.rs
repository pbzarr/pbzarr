//! Import: bulk-write per-base data from `ValueReader` sources into root
//! Dataset arrays via a `crossbeam-channel` worker pool.
//!
//! The engine is format-agnostic: it drives any `ValueReader`. Format readers
//! (d4, bigWig, BED, BAM/CRAM) live in their own crates and start runs through
//! [`Import`].

mod builder;
mod config;
pub(crate) mod destination;
mod engine;
pub mod progress;
mod routing;
mod source;
mod target;

pub use builder::{Import, ImportBuilder};
pub use config::{Config, ConfigBuilder, ProgressSink, Report};
pub use routing::{ImportRouting, SourceAxis, TrackTarget};
pub use source::Source;
pub use target::{Axes, Target, store_reference};
