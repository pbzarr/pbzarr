//! D4-specific import glue. Wires `D4Reader` into the import engine.

use log::{debug, info};
use pbzarr::PbzError;
use pbzarr::Result;
use pbzarr::import::{Axes, Config, Report, Source, Target};
use pbzarr::io::Dtype;

use super::reader::D4Reader;

/// Bulk-import one or more d4 files into a new `int32` track.
///
/// Sizes the column axis from the file list. One source with no configured
/// column dimension yields a scalar track; otherwise the sources form a 2D
/// track whose labels come from each source's `column_label` (falling back to
/// its file stem). All source headers must list the same contigs in the same
/// order.
///
/// A Dataset keeps its own reference, which the sources must match.
/// A [`Target::Path`] that does not exist becomes a new store whose
/// reference is the contig list of the sources.
pub fn from_d4<'t>(
    target: impl Into<Target<'t>>,
    track_name: &str,
    sources: &[Source],
    config: Config,
) -> Result<Report> {
    if sources.is_empty() {
        return Err(PbzError::Metadata("d4 import: no sources".into()));
    }

    info!(
        "d4 import: {} source(s) into track {track_name:?}",
        sources.len()
    );
    let readers: Vec<D4Reader> = sources
        .iter()
        .map(|s| {
            debug!("opening d4 source {}", s.path.display());
            D4Reader::open(&s.path).map_err(|e| PbzError::Store(e.to_string()))
        })
        .collect::<Result<_>>()?;

    crate::check_same_contigs("d4", &readers, sources)?;
    // Several sources, or an explicit column dimension, make a 2D track; a
    // single source with neither stays scalar.
    let labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect::<Vec<String>>());
    let track_config = config.track_config(Dtype::I32, None, labels.clone(), "sample");
    let mut axes = Axes::default();
    if let Some(labels) = labels {
        axes = axes.readers_as_columns(labels);
    }
    target.into().import(
        vec![(track_name.to_owned(), track_config)],
        readers,
        axes,
        config,
    )
}
