//! bigWig-specific import glue. Wires `BigWigReader` into the import engine.

use log::{debug, info};
use pbzarr::Genome;
use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Import, PipelineOptions, Report, Source};
use pbzarr::io::Dtype;
use pbzarr::io::ValueReader as _;

use super::reader::BigWigReader;

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// Builds the track's `Genome` from the source headers and sizes the column axis
/// from the file list. One source with no configured column dimension yields a
/// scalar track; otherwise the sources form a 2D track whose labels come from
/// each source's `column_label` (falling back to its file stem). All sources
/// must share a genome (checked by checksum).
///
/// A bigWig has no value at an uncovered base, so the track takes the `float32`
/// default fill of `NaN`: gaps read back as missing data, not as zero.
pub fn from_bigwig(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[Source],
    config: Config,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("bigWig import: no sources".into()));
    }

    info!(
        "bigWig import: {} source(s) into track {track_name:?}",
        sources.len()
    );
    let readers: Vec<BigWigReader> = sources
        .iter()
        .map(|s| {
            debug!("opening bigWig source {}", s.path.display());
            BigWigReader::open(&s.path)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect::<Result<_>>()?;

    let genome = shared_genome(&readers, sources)?;
    // Several sources, or an explicit column dimension, make a 2D track; a
    // single source with neither stays scalar.
    let labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect::<Vec<String>>());
    let track_config = config.track_config(Dtype::F32, None, labels.clone(), "sample");
    let options = PipelineOptions {
        workers: config.workers,
        in_flight_spans: config.in_flight_spans,
        decode_chunks: config.decode_chunks,
        handle_budget: config.handle_budget,
        progress: config.progress.clone(),
    };
    let report = store.create_tracks_with(
        vec![(track_name.to_owned(), genome, track_config)],
        move |tracks| {
            let mut builder = Import::from_readers(readers)?
                .into_track(tracks[0])
                .options(options);
            if let Some(labels) = labels {
                builder = builder.readers_as_columns().expect_column_labels(labels);
            }
            builder.run()
        },
    )?;
    config.scale_tracks(store, &[track_name.to_owned()])?;
    Ok(report)
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
