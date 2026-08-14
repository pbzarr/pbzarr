//! Bulk-import one BED column across one or more sources into a track.

use pbzarr::import::{Config, Report, Source, run_pipeline};
use pbzarr::io::{Dtype, Numeric};
use pbzarr::{Genome, PbzError, PbzStore, Result, TrackConfig};

use super::reader::BedReader;

/// Import the value at file column `column` (absolute, 0-based, `>= 3`) from each
/// source BED into a new track sized to `genome`. One source with no configured
/// column dimension yields a scalar track; otherwise the sources form a 2D
/// track whose `column_dim` defaults to `"sample"` and whose labels come from
/// each source's `column_label` (falling back to the file stem). Every source
/// must be tabix-indexed and share `genome`'s contig names. Uncovered positions
/// become `T::ZERO`.
pub fn from_bed<T>(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[Source],
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
    store.create_tracks_with(
        vec![(track_name.to_owned(), genome, track_config)],
        move |tracks| run_pipeline::<T, _>(tracks[0], readers, &config),
    )
}

fn track_config<T: Numeric>(sources: &[Source], config: &Config) -> TrackConfig {
    let labels = (sources.len() > 1 || config.column_dim.is_some())
        .then(|| sources.iter().map(Source::label).collect());
    config.track_config(T::DTYPE, Some(zero_fill(T::DTYPE)), labels, "sample")
}

/// Zero fill matching `T::ZERO` so all-gap chunks are elided on write.
pub(super) fn zero_fill(dtype: Dtype) -> serde_json::Value {
    match dtype {
        Dtype::F32 | Dtype::F64 => serde_json::json!(0.0),
        Dtype::Bool => serde_json::json!(false),
        _ => serde_json::json!(0),
    }
}
