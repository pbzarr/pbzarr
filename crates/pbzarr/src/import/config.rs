//! Shared import run state: the track-sizing knobs every format entry point
//! consumes, the progress hook, and the run summary.

use std::sync::Arc;

use crate::codec_spec::ExplicitArraySpec;

/// Hook for callers to observe import progress. Implementations must be
/// `Send + Sync` because workers call `tick` concurrently.
pub trait ProgressSink: Send + Sync {
    /// Total bytes the run will write, reported once before any `tick`. The
    /// engine knows the exact figure (positions x columns x element width),
    /// so callers build a sink without sizing it themselves.
    fn set_total(&self, _bytes: u64) {}
    fn tick(&self, _bytes: u64) {}
    fn done(&self) {}
}

/// Configuration shared by the format import entry points
/// (`from_d4`, `from_bigwig`, `from_bed_matrix`, `from_bam`).
pub struct Config {
    /// Number of reader/writer worker threads.
    pub workers: usize,
    /// Position chunk size for the track being imported. Consumed by the
    /// format entry points when they create the track; the engine always
    /// steps by the track's on-disk chunk grid.
    pub chunk_size: Option<usize>,
    /// Column chunk size for the track being imported. Consumed at track
    /// creation, like `chunk_size`.
    pub column_chunk_size: Option<usize>,
    /// Position shard size for the track being imported. Consumed at track
    /// creation, like `chunk_size`; `None` leaves the track unsharded.
    pub shard_size: Option<usize>,
    /// Column shard size for the track being imported. Consumed at track
    /// creation; ignored unless `shard_size` is set.
    pub shard_column_size: Option<usize>,
    /// Column-axis dimension name for a cohort import (several sources). The
    /// axis is generic; the readers default it to `"sample"`, but set this to
    /// `"strand"`, `"context"`, etc. when the columns are not samples. Ignored
    /// for single-source (scalar) imports.
    pub column_dim: Option<String>,
    /// Explicit Zarr codec/chunk-grid metadata for created `values` arrays.
    /// Consumed at track creation; replaces the default pipeline and the
    /// chunk/shard sizing above.
    pub codecs: Option<ExplicitArraySpec>,
    /// Downsampling factors for an opt-in multiscale pyramid, built on every
    /// track the import creates once its base data is published. Empty builds
    /// no pyramid. The scale engine validates the factors (unique, ascending,
    /// `>= 2`).
    pub scales: Vec<u64>,
    /// Optional progress observer.
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 4,
            chunk_size: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            column_dim: None,
            codecs: None,
            scales: Vec::new(),
            progress: None,
        }
    }
}

impl Config {
    /// Build a `TrackConfig` for `dtype`/`fill` (`None` leaves the track's
    /// default fill unset), applying this config's chunk/shard sizing.
    /// `labels` being `Some` is what makes the track 2D (callers decide via
    /// `n_sources > 1 || self.column_dim.is_some()`, so a single source with
    /// an explicit `column_dim` still gets a 1-column track);
    /// `default_column_dim` is the fallback when `column_dim` is unset.
    /// Shared by every format reader's import entry point.
    pub fn track_config(
        &self,
        dtype: crate::io::Dtype,
        fill: Option<serde_json::Value>,
        labels: Option<Vec<String>>,
        default_column_dim: &str,
    ) -> crate::track::TrackConfig {
        let mut cfg = crate::track::TrackConfig::new(dtype);
        if let Some(fill) = fill {
            cfg = cfg.fill_value(fill);
        }
        if let Some(cs) = self.chunk_size {
            cfg = cfg.chunk_size(cs);
        }
        if let Some(ss) = self.shard_size {
            cfg = cfg.shard_size(ss);
        }
        if let Some(scs) = self.shard_column_size {
            cfg = cfg.shard_column_size(scs);
        }
        if let Some(labels) = labels {
            let dim = self.column_dim.as_deref().unwrap_or(default_column_dim);
            cfg = cfg.columns(labels).column_dim(dim);
            if let Some(ccs) = self.column_chunk_size {
                cfg = cfg.column_chunk_size(ccs);
            }
        }
        if let Some(spec) = &self.codecs {
            cfg = cfg.codecs(spec.clone());
        }
        cfg
    }

    /// Build the configured pyramid on each of `tracks`, in order. A no-op
    /// when `scales` is empty. The factors and worker count pass to the scale
    /// engine unchanged; it owns factor validation and publication ordering.
    /// Format entry points call this after their tracks are published.
    pub fn scale_tracks(&self, store: &crate::PbzStore, tracks: &[String]) -> crate::Result<()> {
        if self.scales.is_empty() {
            return Ok(());
        }
        let config = crate::ScaleConfig {
            factors: Some(self.scales.clone()),
            workers: self.workers,
            ..crate::ScaleConfig::default()
        };
        for track in tracks {
            crate::scale(store, track, &config)?;
        }
        Ok(())
    }
}

/// Summary of a finished import.
pub struct Report {
    pub contigs_written: usize,
    pub bytes_written: u64,
    /// Number of buffers written.
    pub tasks_completed: usize,
    /// Number of buffers elided because no reader covered them
    /// (`may_have_data` was false for every overlapping contig).
    pub tasks_skipped: usize,
}
