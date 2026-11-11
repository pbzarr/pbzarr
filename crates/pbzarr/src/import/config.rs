//! Shared import run state: the track-sizing knobs every format entry point
//! consumes, the progress hook, and the run summary.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::codec_spec::ExplicitArraySpec;
use crate::import::progress::make_sink;

/// Hook for callers to observe import progress. Implementations must be
/// `Send + Sync` because workers call `tick` concurrently.
pub trait ProgressSink: Send + Sync {
    /// Total bytes the run covers (positions x columns x element width),
    /// reported once before any `tick`. Callers build a sink without sizing
    /// it themselves.
    fn set_total(&self, _bytes: u64) {}
    /// Bytes of one closed buffer, written or skipped. The ticks of a run
    /// sum to the `set_total` figure.
    fn tick(&self, _bytes: u64) {}
    fn done(&self) {}
}

/// Configuration of one import run, shared by the format import entry points
/// (`from_d4`, `from_bigwig`, `from_bed_matrix`, `from_bam`) and the engine.
/// The engine reads `workers`, `decode_chunks`, `progress` and `cancel`; the entry
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
    /// Positions per inner chunk of the tracks the import creates. The
    /// engine always steps by the track's on-disk chunk grid. Default:
    /// [`DEFAULT_CHUNK`](crate::DEFAULT_CHUNK) (65,536).
    pub chunk: usize,
    /// Inner chunks per shard along positions of the created tracks; `None`
    /// leaves them unsharded. Default:
    /// `Some(`[`DEFAULT_SHARD`](crate::DEFAULT_SHARD)`)` (1,024).
    pub shard: Option<usize>,
    /// Columns per shard of the created 2D tracks. Default:
    /// [`DEFAULT_SHARD_COLUMNS`](crate::DEFAULT_SHARD_COLUMNS) (16).
    pub shard_columns: usize,
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
    /// When another thread sets this flag, the engine stops before its next
    /// piece and the run fails with `PbzError::Cancelled`.
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workers: 4,
            decode_chunks: 0,
            chunk: crate::DEFAULT_CHUNK,
            shard: Some(crate::DEFAULT_SHARD),
            shard_columns: crate::DEFAULT_SHARD_COLUMNS,
            column_dim: None,
            codecs: None,
            progress: None,
            cancel: None,
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
        fill: Option<crate::Fill>,
        labels: Option<Vec<String>>,
        default_column_dim: &str,
    ) -> crate::TrackConfig {
        let mut cfg = crate::TrackConfig::new(dtype)
            .chunk(self.chunk)
            .shard_columns(self.shard_columns);
        cfg.shard = self.shard;
        if let Some(fill) = fill {
            cfg = cfg.fill(fill);
        }
        if let Some(labels) = labels {
            let dim = self.column_dim.as_deref().unwrap_or(default_column_dim);
            cfg = cfg.columns(labels).column_dim(dim);
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
    pub chunk: Option<usize>,
    pub shard: Option<usize>,
    pub shard_columns: Option<usize>,
    pub column_dim: Option<String>,
    pub codecs: Option<ExplicitArraySpec>,
    /// Label of the progress display; `None` shows no progress.
    pub progress: Option<String>,
}

impl ConfigBuilder {
    /// The first layout knob that is set. A `codecs` spec replaces the
    /// layout, so a caller refuses them together.
    pub fn layout_knob_set(&self) -> Option<&'static str> {
        [
            ("chunk", self.chunk),
            ("shard", self.shard),
            ("shard_columns", self.shard_columns),
        ]
        .into_iter()
        .find_map(|(name, size)| size.map(|_| name))
    }

    pub fn build(self) -> Config {
        let default = Config::default();
        Config {
            workers: self.workers.unwrap_or(default.workers),
            decode_chunks: self.decode_chunks.unwrap_or(default.decode_chunks),
            chunk: self.chunk.unwrap_or(default.chunk),
            shard: self.shard.or(default.shard),
            shard_columns: self.shard_columns.unwrap_or(default.shard_columns),
            column_dim: self.column_dim,
            codecs: self.codecs,
            progress: self.progress.as_deref().map(make_sink),
            cancel: None,
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
    fn builder_keeps_defaults_and_names_the_first_layout_knob() {
        let builder = ConfigBuilder {
            shard: Some(8),
            shard_columns: Some(2),
            ..ConfigBuilder::default()
        };
        assert_eq!(builder.layout_knob_set(), Some("shard"));
        let config = builder.build();
        assert_eq!(config.workers, Config::default().workers);
        assert_eq!(config.decode_chunks, 0);
        assert_eq!(config.chunk, crate::DEFAULT_CHUNK);
        assert_eq!(config.shard, Some(8));
        assert_eq!(config.shard_columns, 2);
        assert!(config.progress.is_none());
    }
}
