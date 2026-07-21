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

mod progress;

create_exception!(_native, PbzError, PyRuntimeError);

/// Total bytes an import will write: positions x sources x element size. This
/// matches the pipeline's per-chunk byte accounting, so a progress bar sized to
/// it fills to exactly 100%. `sum_len` (ΣL) comes from the source headers, since
/// the genome now belongs to the track the readers create, not the store.
fn total_bytes(sum_len: u64, n_sources: usize, elem_size: usize) -> u64 {
    sum_len * n_sources as u64 * elem_size as u64
}

/// Bulk-import one or more d4 files into a new `int32` track.
///
/// The track is created from the source headers; it must NOT already exist.
/// d4 stores depths as `int32` natively, so import is zero-conversion.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, workers=None, chunk_size=None, column_chunk_size=None, progress=false))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_d4(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    progress: bool,
) -> PyResult<()> {
    py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let d4_sources: Vec<D4Source> = sources
            .iter()
            .map(|(path, sample_label)| D4Source {
                path: PathBuf::from(path),
                sample_label: sample_label.clone(),
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
        if progress && let Some((first, _)) = sources.first() {
            let sum_len: u64 = pbzarr_readers::d4::contigs(first)
                .map_err(|e| PbzError::new_err(format!("{e}")))?
                .iter()
                .map(|(_, len)| *len)
                .sum();
            let total = total_bytes(sum_len, d4_sources.len(), std::mem::size_of::<i32>());
            config.progress = Some(progress::make_sink(&track, total));
        }
        rs_from_d4(&mut store, &track, &d4_sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))?;
        Ok(())
    })
}

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// The track is created from the source headers; it must NOT already exist.
/// bigWig stores values as `float32` natively. Positions not covered by any
/// bigWig become `0.0`, and the track is created with a `0.0` fill value.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, workers=None, chunk_size=None, column_chunk_size=None, progress=false))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bigwig(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    progress: bool,
) -> PyResult<()> {
    py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let bw_sources: Vec<BigWigSource> = sources
            .iter()
            .map(|(path, sample_label)| BigWigSource {
                path: PathBuf::from(path),
                sample_label: sample_label.clone(),
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
        if progress && let Some((first, _)) = sources.first() {
            let sum_len: u64 = pbzarr_readers::bigwig::contigs(first)
                .map_err(|e| PbzError::new_err(format!("{e}")))?
                .iter()
                .map(|(_, len)| *len)
                .sum();
            let total = total_bytes(sum_len, bw_sources.len(), std::mem::size_of::<f32>());
            config.progress = Some(progress::make_sink(&track, total));
        }
        rs_from_bigwig(&mut store, &track, &bw_sources, config)
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
