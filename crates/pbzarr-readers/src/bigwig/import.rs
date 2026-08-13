//! bigWig-specific import glue. Wires `BigWigReader` into the generic pipeline.

use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Report, Source, run_pipeline};
use pbzarr::io::{Dtype, ValueReader};
use pbzarr::{Genome, TrackConfig};

use super::reader::BigWigReader;

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// Builds the track's `Genome` from the source headers and sizes the column axis
/// from the file list. One source with no configured column dimension yields a
/// scalar track;
/// otherwise the sources form a 2D track whose labels come from each source's
/// `column_label` (falling back to its file stem). All sources must share a
/// genome (checked by checksum).
///
/// `BigWigReader` maps uncovered positions to `0.0`, so the track is created
/// with a `0.0` fill value; all-gap chunks then equal the fill and are elided.
pub fn from_bigwig(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[Source],
    config: Config,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("bigWig import: no sources".into()));
    }

    let readers: Vec<BigWigReader> = sources
        .iter()
        .map(|s| {
            BigWigReader::open(&s.path)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect::<Result<_>>()?;

    let genome = shared_genome(&readers, sources)?;
    let track_config = track_config(sources, &config);
    store.create_tracks_with(
        vec![(track_name.to_owned(), genome, track_config)],
        move |tracks| run_pipeline::<f32, _>(tracks[0], readers, &config),
    )
}

/// The genome shared by every source, taken from the first and required to
/// match the rest by checksum (all files must describe the same reference).
fn shared_genome(readers: &[BigWigReader], sources: &[Source]) -> Result<Genome> {
    let genome = readers[0].contigs().clone();
    let checksum = genome.checksum();
    for (reader, source) in readers.iter().zip(sources).skip(1) {
        if reader.contigs().checksum() != checksum {
            return Err(PbzError::Metadata(format!(
                "bigWig import: {} genome differs from {}",
                source.path.display(),
                sources[0].path.display()
            )));
        }
    }
    Ok(genome)
}

fn track_config(sources: &[Source], config: &Config) -> TrackConfig {
    let labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect());
    config.track_config(Dtype::F32, Some(serde_json::json!(0.0)), labels, "sample")
}
