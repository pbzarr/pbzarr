//! D4-specific import glue. Wires `D4Reader` into the generic pipeline.

use std::path::PathBuf;

use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Report, run_pipeline};
use pbzarr::io::{Dtype, ValueReader};
use pbzarr::{Genome, TrackConfig};

use super::reader::D4Reader;

#[derive(Debug, Clone)]
pub struct D4Source {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Bulk-import one or more d4 files into a new `int32` track.
///
/// Builds the track's `Genome` from the source headers, sizes the column axis
/// from the file list, and creates the track before running the pipeline. One
/// source yields a scalar track; several yield a cohort track whose column
/// labels come from each source's `sample_label` (falling back to its file
/// stem). All sources must share a genome (checked by checksum).
pub fn from_d4(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[D4Source],
    config: Config,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("d4 import: no sources".into()));
    }

    let readers: Vec<D4Reader> = sources
        .iter()
        .map(|s| {
            D4Reader::open(&s.path)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect::<Result<_>>()?;

    let genome = shared_genome(&readers, sources)?;
    let track_config = track_config(sources, &config);
    let track = store.create_track(track_name, genome, track_config)?;

    run_pipeline::<i32, _>(track, readers, &config)
}

/// The genome shared by every source, taken from the first and required to
/// match the rest by checksum (all files must describe the same reference).
fn shared_genome(readers: &[D4Reader], sources: &[D4Source]) -> Result<Genome> {
    let genome = readers[0].contigs().clone();
    let checksum = genome.checksum();
    for (reader, source) in readers.iter().zip(sources).skip(1) {
        if reader.contigs().checksum() != checksum {
            return Err(PbzError::Metadata(format!(
                "d4 import: {} genome differs from {}",
                source.path.display(),
                sources[0].path.display()
            )));
        }
    }
    Ok(genome)
}

fn track_config(sources: &[D4Source], config: &Config) -> TrackConfig {
    let mut cfg = TrackConfig::new(Dtype::I32);
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

fn column_label(source: &D4Source) -> String {
    source.sample_label.clone().unwrap_or_else(|| {
        source
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.path.to_string_lossy().into_owned())
    })
}
