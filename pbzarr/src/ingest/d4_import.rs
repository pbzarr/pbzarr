//! D4-specific ingest glue. Wires `D4Reader` into the generic pipeline.

use std::path::PathBuf;

use crate::error::PbzError;
use crate::ingest::pipeline::{run_pipeline, ImportConfig, ImportReport};
use crate::io::{D4Reader, Dtype};
use crate::PbzStore;
use crate::Result;

#[derive(Debug, Clone)]
pub struct D4Source {
    pub path: PathBuf,
    pub sample_label: Option<String>,
}

/// Bulk-import one or more d4 files into an existing track.
///
/// The track MUST already exist (created via `PbzStore::create_track`). Its
/// dtype MUST be `uint32` — d4 ingest only supports that at v0. `sources.len()`
/// MUST equal the track's column count for cohort tracks, or be exactly 1 for
/// scalar tracks.
pub fn import_d4(
    store: &mut PbzStore,
    track_name: &str,
    sources: &[D4Source],
    config: ImportConfig,
) -> Result<ImportReport> {
    let track = store.track(track_name).ok_or_else(|| PbzError::TrackNotFound {
        name: track_name.to_owned(),
        available: store.track_names().map(|s| s.to_owned()).collect(),
    })?;

    if track.dtype() != Dtype::U32 {
        return Err(PbzError::InvalidDtype {
            dtype: format!(
                "d4 import requires uint32 track; track {track_name:?} is {}",
                track.dtype()
            ),
        });
    }

    let expected_n = if track.rank() == 1 {
        1
    } else {
        track.columns_count()
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
            D4Reader::open(&s.path).map_err(|e| {
                PbzError::Store(format!("open {}: {e}", s.path.display()))
            })
        })
        .collect();
    let readers = readers?;

    run_pipeline::<u32, _>(track, readers, &config)
}
