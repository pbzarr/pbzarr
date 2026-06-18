//! bigWig-specific import glue. Wires `BigWigReader` into the generic pipeline.

use std::path::PathBuf;

use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Report, run_pipeline};
use pbzarr::io::Dtype;

use super::reader::BigWigReader;

#[derive(Debug, Clone)]
pub struct BigWigSource {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Bulk-import one or more bigWig files into an existing track.
///
/// The track MUST already exist (created via `PbzStore::create_track`). Its
/// dtype MUST be `float32` — bigWig stores values as f32 natively, so import is
/// zero-conversion at the per-position level. `sources.len()` MUST equal the
/// track's column count for cohort tracks, or be exactly 1 for scalar tracks.
///
/// Positions not covered by a bigWig become `0` ("no coverage" = 0 for
/// coverage/percent tracks). Create the track with `fill_value = 0.0` so all-gap
/// chunks equal the fill value and are elided on write.
pub fn from_bigwig(
    store: &PbzStore,
    track_name: &str,
    sources: &[BigWigSource],
    config: Config,
) -> Result<Report> {
    let track = store
        .track(track_name)
        .ok_or_else(|| PbzError::TrackNotFound {
            name: track_name.to_owned(),
            available: store.track_names().map(|s| s.to_owned()).collect(),
        })?;

    if track.dtype() != Dtype::F32 {
        return Err(PbzError::InvalidDtype {
            dtype: format!(
                "bigWig import requires float32 track; track {track_name:?} is {}",
                track.dtype()
            ),
        });
    }

    let expected_n = if track.rank() == 1 {
        1
    } else {
        track.columns_count()?
    };
    if sources.len() != expected_n {
        return Err(PbzError::Metadata(format!(
            "bigWig import: track {track_name:?} expects {expected_n} source(s); got {}",
            sources.len()
        )));
    }

    let readers: Result<Vec<BigWigReader>> = sources
        .iter()
        .map(|s| {
            BigWigReader::open(&s.path)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect();
    let readers = readers?;

    run_pipeline::<f32, _>(track, readers, &config)
}
