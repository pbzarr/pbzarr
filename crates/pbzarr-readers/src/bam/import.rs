//! BAM/CRAM-specific import glue. Wires `BamReader` into the import engine.

use std::path::PathBuf;

use log::{debug, info, warn};
use pbzarr::Genome;
use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Import, PipelineOptions, Report, Source};
use pbzarr::io::Dtype;
use pbzarr::io::ValueReader as _;

use super::reader::BamReader;
use super::walk::{FIELDS, ImportMode};
use super::{DepthFilter, OverlapMode};

/// Bulk-import per-base depth or composition counts from one or more
/// BAM/CRAM files.
///
/// `Depth` writes a single `track_name` track. `Composition` writes one track
/// per [`FIELDS`] entry: index 0 (`"depth"`) keeps the bare `track_name`, the
/// rest are named `"{track_name}_{field}"` (e.g. `"{track_name}_ins"`).
///
/// Every source's `Genome` is built from its own header and must
/// checksum-match the rest (mirrors `from_d4`/`from_bigwig`). CRAM sources
/// need `reference`; BAM sources ignore it. One source with no configured
/// column dimension yields scalar tracks; otherwise the sources become the
/// columns of cohort tracks, whose dimension name comes from
/// `config.column_dim` and defaults to `"sample"`.
pub fn from_bam(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[Source],
    mode: ImportMode,
    filter: DepthFilter,
    reference: Option<PathBuf>,
    config: Config,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("bam import: no sources".into()));
    }

    info!(
        "bam import: {} source(s) into track {track_name:?}, mode {mode:?}",
        sources.len()
    );
    debug!("bam import filter: {filter:?}, reference {reference:?}");
    if reference.is_none()
        && let Some(cram) = sources
            .iter()
            .find(|s| s.path.extension().is_some_and(|ext| ext == "cram"))
    {
        warn!(
            "{} looks like CRAM but no reference FASTA was given; decoding will fail unless the file embeds its reference",
            cram.path.display()
        );
    }
    let readers: Vec<BamReader> = sources
        .iter()
        .map(|s| {
            debug!("opening alignment source {}", s.path.display());
            BamReader::open(&s.path, reference.as_deref(), mode, filter)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect::<Result<_>>()?;

    let genome = shared_genome(&readers, sources)?;
    debug!(
        "bam import genome: {} contig(s), {} positions",
        genome.contigs().len(),
        genome.contigs().iter().map(|c| c.length).sum::<u64>()
    );

    let names = track_names(track_name, mode);
    info!("bam import writes track(s) {names:?}");
    let source_paths = sources
        .iter()
        .map(|s| s.path.display().to_string())
        .collect::<Vec<_>>()
        .join(",");
    // Several sources, or an explicit column dimension, make 2D tracks; a
    // single source with neither stays scalar.
    let column_labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect::<Vec<_>>());
    let specs = names
        .iter()
        .map(|name| {
            let cfg = config
                .track_config(
                    Dtype::I32,
                    Some(serde_json::json!(0)),
                    column_labels.clone(),
                    "sample",
                )
                .description(describe_track(name, track_name, &filter))
                .source(source_paths.clone());
            (name.clone(), genome.clone(), cfg)
        })
        .collect();

    let options = PipelineOptions {
        workers: config.workers,
        in_flight_spans: config.in_flight_spans,
        progress: config.progress.clone(),
    };
    let report = store.create_tracks_with(specs, move |tracks| {
        let builder = Import::from_readers(readers)?;
        let mut builder = match mode {
            ImportMode::Depth => builder.into_track(tracks[0]),
            ImportMode::Composition => builder.into_tracks(tracks).fields_as_tracks(),
        };
        if let Some(labels) = column_labels {
            builder = builder.readers_as_columns().expect_column_labels(labels);
        }
        builder.options(options).run()
    })?;
    config.scale_tracks(store, &names)?;
    Ok(report)
}

/// The genome shared by every source, taken from the first and required to
/// match the rest by checksum (all files must describe the same reference).
fn shared_genome(readers: &[BamReader], sources: &[Source]) -> Result<Genome> {
    let genome = readers
        .first()
        .ok_or_else(|| PbzError::Metadata("bam import: no readers to derive a genome from".into()))?
        .contigs()
        .clone();
    let checksum = genome.checksum();
    for (reader, source) in readers.iter().zip(sources).skip(1) {
        if reader.contigs().checksum() != checksum {
            return Err(PbzError::Metadata(format!(
                "bam import: {} genome differs from {}",
                source.path.display(),
                sources[0].path.display()
            )));
        }
    }
    Ok(genome)
}

/// Spec-required `description`: field name plus the filter settings that
/// shaped it, so a track's counting rules travel with the store instead of
/// living only in the caller's import invocation.
fn describe_track(name: &str, track_name: &str, filter: &DepthFilter) -> String {
    // Composition tracks are named `{track_name}_{field}` (see `track_names`);
    // the description names the field, not the store-level track. In Depth
    // mode `name == track_name`, so the field is always "depth" -- there's
    // no prefix to strip, deliberately, not a fallback for a failed strip.
    let field = if name == track_name {
        "depth"
    } else {
        name.strip_prefix(&format!("{track_name}_"))
            .unwrap_or("depth")
    };
    let overlap = match filter.overlap {
        OverlapMode::None => "none",
        OverlapMode::ProperOnly => "proper",
        OverlapMode::All => "all",
    };
    let deletions = if filter.count_deletions {
        "counted"
    } else {
        "uncounted"
    };
    format!(
        "{field} (mapq>={}, bq>={}, flags&{} excluded, overlap={overlap}, deletions {deletions})",
        filter.min_mapq, filter.min_bq, filter.exclude_flags
    )
}

fn track_names(track_name: &str, mode: ImportMode) -> Vec<String> {
    match mode {
        ImportMode::Depth => vec![track_name.to_owned()],
        ImportMode::Composition => FIELDS
            .iter()
            .enumerate()
            .map(|(i, field)| {
                if i == 0 {
                    track_name.to_owned()
                } else {
                    format!("{track_name}_{field}")
                }
            })
            .collect(),
    }
}
