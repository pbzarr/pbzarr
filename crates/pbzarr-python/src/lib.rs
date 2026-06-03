//! PyO3 bindings for pbzarr. Single function: `import_d4`.

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pbzarr::PbzStore;
use pbzarr::import::Config;
use pbzarr_readers::{D4Source, from_d4 as rs_from_d4};

create_exception!(_native, PbzError, PyRuntimeError);

/// Bulk-import one or more d4 files into an existing track.
///
/// The track MUST already exist (created via `create_track`). dtype is
/// read from the track metadata and currently MUST be `int32`.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, workers=None, chunk_size=None, column_chunk_size=None))]
fn import_d4(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
) -> PyResult<()> {
    py.allow_threads(|| {
        let store = PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let sources: Vec<D4Source> = sources
            .into_iter()
            .map(|(path, sample_label)| D4Source {
                path: PathBuf::from(path),
                sample_label,
            })
            .collect();
        let mut config = Config::default();
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(c) = column_chunk_size {
            config.column_chunk_size = Some(c);
        }
        rs_from_d4(&store, &track, &sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))?;
        Ok(())
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PbzError", m.py().get_type::<PbzError>())?;
    m.add_function(wrap_pyfunction!(import_d4, m)?)?;
    Ok(())
}
