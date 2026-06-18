//! PyO3 bindings for pbzarr: `import_d4` and `import_bigwig`.

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pbzarr::PbzStore;
use pbzarr::import::Config;
use pbzarr_readers::{
    BigWigSource, D4Source, from_bigwig as rs_from_bigwig, from_d4 as rs_from_d4,
};

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

/// Bulk-import one or more bigWig files into an existing track.
///
/// The track MUST already exist (created via `create_track`). dtype is read
/// from the track metadata and currently MUST be `float32`. Positions not
/// covered by a bigWig become `NaN`.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, workers=None, chunk_size=None, column_chunk_size=None))]
fn import_bigwig(
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
        let sources: Vec<BigWigSource> = sources
            .into_iter()
            .map(|(path, sample_label)| BigWigSource {
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
        rs_from_bigwig(&store, &track, &sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))?;
        Ok(())
    })
}

/// Read a d4 file's contig list from its header.
///
/// Returns `(name, length)` pairs in file order, sizing a store directly from
/// the source without pyd4 or an external d4tools call.
#[pyfunction]
fn d4_contigs(py: Python<'_>, path: String) -> PyResult<Vec<(String, u64)>> {
    py.allow_threads(|| {
        pbzarr_readers::d4::contigs(&path).map_err(|e| PbzError::new_err(format!("{e}")))
    })
}

/// Read a bigWig file's contig list from its header.
///
/// Returns `(name, length)` pairs in the file's chrom order, sizing a store
/// directly from the source without pybigtools.
#[pyfunction]
fn bigwig_contigs(py: Python<'_>, path: String) -> PyResult<Vec<(String, u64)>> {
    py.allow_threads(|| {
        pbzarr_readers::bigwig::contigs(&path).map_err(|e| PbzError::new_err(format!("{e}")))
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PbzError", m.py().get_type::<PbzError>())?;
    m.add_function(wrap_pyfunction!(import_d4, m)?)?;
    m.add_function(wrap_pyfunction!(import_bigwig, m)?)?;
    m.add_function(wrap_pyfunction!(d4_contigs, m)?)?;
    m.add_function(wrap_pyfunction!(bigwig_contigs, m)?)?;
    Ok(())
}
