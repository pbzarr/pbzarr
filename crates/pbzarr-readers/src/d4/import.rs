//! D4-specific import glue. Wires `D4Reader` into the generic pipeline.

use std::path::PathBuf;

use pbzarr::PbzError;
use pbzarr::PbzStore;
use pbzarr::Result;
use pbzarr::import::{Config, Report, run_pipeline};
use pbzarr::io::Dtype;

use super::reader::D4Reader;

#[derive(Debug, Clone)]
pub struct D4Source {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Bulk-import one or more d4 files into an existing track.
///
/// The track MUST already exist (created via `PbzStore::create_track`). Its
/// dtype MUST be `int32` — d4 stores depths as i32 natively, so import is
/// zero-conversion at the per-position level. `sources.len()` MUST equal the
/// track's column count for cohort tracks, or be exactly 1 for scalar tracks.
pub fn from_d4(
    store: &PbzStore,
    track_name: &str,
    sources: &[D4Source],
    config: Config,
) -> Result<Report> {
    let track = store
        .track(track_name)
        .ok_or_else(|| PbzError::TrackNotFound {
            name: track_name.to_owned(),
            available: store.track_names().map(|s| s.to_owned()).collect(),
        })?;

    if track.dtype() != Dtype::I32 {
        return Err(PbzError::InvalidDtype {
            dtype: format!(
                "d4 import requires int32 track; track {track_name:?} is {}",
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
            "d4 import: track {track_name:?} expects {expected_n} source(s); got {}",
            sources.len()
        )));
    }

    let readers: Result<Vec<D4Reader>> = sources
        .iter()
        .map(|s| {
            D4Reader::open(&s.path)
                .map_err(|e| PbzError::Store(format!("open {}: {e}", s.path.display())))
        })
        .collect();
    let readers = readers?;

    run_pipeline::<i32, _>(track, readers, &config)
}
