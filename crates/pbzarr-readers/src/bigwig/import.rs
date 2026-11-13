//! bigWig-specific import glue. Wires `BigWigReader` into the import engine.

use log::{debug, info};
use pbzarr::import::{Axes, Config, Report, Source, Target};
use pbzarr::io::Dtype;
use pbzarr::{Fill, PbzError, Result};

use super::reader::BigWigReader;

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// Sizes the column axis from the file list. One source with no configured
/// column dimension yields a scalar track; otherwise the sources form a 2D
/// track whose labels come from each source's `column_label` (falling back to
/// its file name without the format extension). All source headers must list the same contigs in the same
/// order.
///
/// A Dataset keeps its own reference, which the sources must match.
/// A [`Target::Path`] that does not exist becomes a new store whose
/// reference is the contig list of the sources.
///
/// `fill` is the value of bases that no source interval covers. `None` gives
/// [`Fill::Na`] on a new track: a gap reads as NaN, which marks a missing
/// value. An append with `None` keeps the fill of the stored track, and an
/// append with another fill than the stored one is refused.
pub fn from_bigwig<'t>(
    target: impl Into<Target<'t>>,
    track_name: &str,
    sources: &[Source],
    fill: Option<Fill>,
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
            BigWigReader::open(&s.path).map_err(|e| crate::open_failed(&s.path, e))
        })
        .collect::<Result<_>>()?;

    crate::check_same_contigs("bigWig", &readers, sources)?;
    // Several sources, or an explicit column dimension, make a 2D track; a
    // single source with neither stays scalar.
    let labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect::<Vec<String>>());
    let target = target.into();
    let fill = match target {
        Target::Append(_) => fill,
        _ => Some(fill.unwrap_or(Fill::Na)),
    };
    let track_config = config.track_config(Dtype::F32, fill, labels.clone(), "sample");
    let mut axes = Axes::default();
    if let Some(labels) = labels {
        axes = axes.readers_as_columns(labels);
    }
    target.import(
        vec![(track_name.to_owned(), track_config)],
        readers,
        axes,
        config,
    )
}
