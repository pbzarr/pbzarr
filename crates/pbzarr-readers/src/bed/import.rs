//! Bulk-import one BED column across N sources into a 2D track.

use std::path::PathBuf;

use pbzarr::import::{Config, Report, run_pipeline};
use pbzarr::io::{Dtype, Numeric};
use pbzarr::{Genome, PbzError, PbzStore, Result, TrackConfig};

use super::reader::BedReader;

pub struct BedSource {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Import the value at file column `column` (absolute, 0-based, `>= 3`) from each
/// source BED into a new track sized to `genome`. One source yields a scalar
/// track; several yield a 2D track whose `column_dim` comes from
/// `Config::column_dim` (default `"sample"`) and whose labels come from each
/// source's `sample_label` (falling back to the file stem). Every source must
/// be tabix-indexed and share `genome`'s contig names. Uncovered positions
/// become `T::ZERO`.
pub fn from_bed<T>(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[BedSource],
    column: usize,
    genome: Genome,
    config: Config,
) -> Result<Report>
where
    T: Numeric + std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if sources.is_empty() {
        return Err(PbzError::Metadata("bed import: no sources".into()));
    }

    let readers: Vec<BedReader<T>> = sources
        .iter()
        .map(|s| {
            BedReader::open(&s.path, column, genome.clone())
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect::<Result<_>>()?;

    let track_config = track_config::<T>(sources, &config);
    let track = store.create_track(track_name, genome, track_config)?;

    run_pipeline::<T, _>(track, readers, &config)
}

fn track_config<T: Numeric>(sources: &[BedSource], config: &Config) -> TrackConfig {
    let mut cfg = TrackConfig::new(T::DTYPE).fill_value(zero_fill(T::DTYPE));
    if let Some(cs) = config.chunk_size {
        cfg = cfg.chunk_size(cs);
    }
    if let Some(ss) = config.shard_size {
        cfg = cfg.shard_size(ss);
    }
    if let Some(scs) = config.shard_column_size {
        cfg = cfg.shard_column_size(scs);
    }
    if sources.len() > 1 {
        let labels: Vec<String> = sources.iter().map(column_label).collect();
        let dim = config.column_dim.as_deref().unwrap_or("sample");
        cfg = cfg.columns(labels).column_dim(dim);
        if let Some(ccs) = config.column_chunk_size {
            cfg = cfg.column_chunk_size(ccs);
        }
    }
    cfg
}

/// Zero fill matching `T::ZERO` so all-gap chunks are elided on write.
pub(super) fn zero_fill(dtype: Dtype) -> serde_json::Value {
    match dtype {
        Dtype::F32 | Dtype::F64 => serde_json::json!(0.0),
        Dtype::Bool => serde_json::json!(false),
        _ => serde_json::json!(0),
    }
}

fn column_label(source: &BedSource) -> String {
    source.sample_label.clone().unwrap_or_else(|| {
        source
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.path.to_string_lossy().into_owned())
    })
}
