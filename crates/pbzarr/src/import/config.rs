//! Shared import run state: the track-sizing knobs every format entry point
//! consumes, the progress hook, and the run summary.

use std::sync::Arc;

use crate::codec_spec::ExplicitArraySpec;
use crate::import::progress::make_sink;

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

/// Configuration of one import run, shared by the format import entry points
/// (`from_d4`, `from_bigwig`, `from_bed_matrix`, `from_bam`) and the engine.
/// The engine reads `workers`, `decode_chunks` and `progress`; the entry
/// points read the track sizing fields when they create the tracks.
pub struct Config {
    /// Worker thread count: the size of the one pool that runs both source
    /// decoding and zarrs encoding, so at most this many threads are busy.
    pub workers: usize,
    /// Inner position chunks each reader decodes per piece. `0` is auto: as
    /// many chunks as fit in 4M positions, fewer when that leaves workers
    /// short of pieces. An explicit value is clamped to `[1, chunk count]`,
    /// and either is snapped to the shard grid of sharded tracks.
    pub decode_chunks: usize,
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
    /// Explicit Zarr codec/chunk-grid metadata for the created track arrays.
    /// Consumed at track creation; replaces the default pipeline and the
    /// chunk/shard sizing above.
    pub codecs: Option<ExplicitArraySpec>,
    /// Optional progress observer.
    pub progress: Option<Arc<dyn ProgressSink>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 4,
            decode_chunks: 0,
            chunk_size: None,
            column_chunk_size: None,
            shard_size: None,
            shard_column_size: None,
            column_dim: None,
            codecs: None,
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
    ) -> crate::TrackConfig {
        let mut cfg = crate::TrackConfig::new(dtype);
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
}

/// A [`Config`] from optional settings, as the CLI and the Python bindings
/// take them. A `None` field keeps the `Config` default.
#[derive(Default)]
pub struct ConfigBuilder {
    pub workers: Option<usize>,
    pub decode_chunks: Option<usize>,
    pub chunk_size: Option<usize>,
    pub column_chunk_size: Option<usize>,
    pub shard_size: Option<usize>,
    pub shard_column_size: Option<usize>,
    pub column_dim: Option<String>,
    pub codecs: Option<ExplicitArraySpec>,
    /// Label of the progress display; `None` shows no progress.
    pub progress: Option<String>,
}

impl ConfigBuilder {
    /// The first chunk or shard size that is set. A `codecs` spec replaces
    /// these sizes, so a caller refuses them together.
    pub fn grid_size_set(&self) -> Option<&'static str> {
        [
            ("chunk_size", self.chunk_size),
            ("column_chunk_size", self.column_chunk_size),
            ("shard_size", self.shard_size),
            ("shard_column_size", self.shard_column_size),
        ]
        .into_iter()
        .find_map(|(name, size)| size.map(|_| name))
    }

    pub fn build(self) -> Config {
        let default = Config::default();
        Config {
            workers: self.workers.unwrap_or(default.workers),
            decode_chunks: self.decode_chunks.unwrap_or(default.decode_chunks),
            chunk_size: self.chunk_size,
            column_chunk_size: self.column_chunk_size,
            shard_size: self.shard_size,
            shard_column_size: self.shard_column_size,
            column_dim: self.column_dim,
            codecs: self.codecs,
            progress: self.progress.as_deref().map(make_sink),
        }
    }
}

/// Summary of a finished import.
pub struct Report {
    /// Positions the import covered, summed over every contig.
    pub positions: u64,
    /// Chunk buffers written.
    pub chunks_written: usize,
    /// Chunk buffers skipped because no reader covered them.
    pub chunks_skipped: usize,
    pub bytes_written: u64,
    /// Wall-clock duration of the engine run.
    pub wall_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_keeps_defaults_and_names_the_first_grid_size() {
        let builder = ConfigBuilder {
            shard_size: Some(8),
            column_chunk_size: Some(2),
            ..ConfigBuilder::default()
        };
        assert_eq!(builder.grid_size_set(), Some("column_chunk_size"));
        let config = builder.build();
        assert_eq!(config.workers, Config::default().workers);
        assert_eq!(config.decode_chunks, 0);
        assert_eq!(config.shard_size, Some(8));
        assert!(config.progress.is_none());
    }
}
