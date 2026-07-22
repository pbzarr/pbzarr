//! PyO3 bindings for pbzarr: `import_d4`, `import_bigwig`, and `import_bed`.

use std::path::{Path, PathBuf};

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pbzarr::Genome;
use pbzarr::PbzStore;
use pbzarr::import::Config;
use pbzarr::io::Dtype;
use pbzarr_readers::{
    BedColumnSpec, BedSchema, BedSource, BigWigSource, D4Source, column_index_by_name,
    from_bed as rs_from_bed, from_bed_multi as rs_from_bed_multi, from_bigwig as rs_from_bigwig,
    from_d4 as rs_from_d4,
};

/// Bytes per element for progress accounting.
fn dtype_bytes(dt: Dtype) -> u64 {
    match dt {
        Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        Dtype::U16 | Dtype::I16 => 2,
        Dtype::U32 | Dtype::I32 | Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

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

/// Import one column from N bgzipped, tabix-indexed BED files into a track.
///
/// `column` is a header name; `dtype` is one of "int32" | "float32" | "bool".
/// `genome` is a .fai / chrom.sizes path (BED files carry no contig lengths).
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, column, dtype, genome, workers=None, chunk_size=None, column_chunk_size=None, progress=false))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bed(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    column: String,
    dtype: String,
    genome: String,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    progress: bool,
) -> PyResult<()> {
    py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let bed_sources: Vec<BedSource> = sources
            .iter()
            .map(|(path, sample_label)| BedSource {
                path: PathBuf::from(path),
                sample_label: sample_label.clone(),
            })
            .collect();
        let (first, _) = bed_sources
            .first()
            .map(|s| (&s.path, ()))
            .ok_or_else(|| PbzError::new_err("bed import: no sources"))?;
        let column_idx =
            column_index_by_name(first, &column).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let genome = Genome::from_fai(&genome).map_err(|e| PbzError::new_err(format!("{e}")))?;

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
        if progress {
            let sum_len: u64 = genome.contigs().iter().map(|c| c.length).sum();
            let elem = match dtype.as_str() {
                "int32" => std::mem::size_of::<i32>(),
                "float32" => std::mem::size_of::<f32>(),
                "bool" => std::mem::size_of::<bool>(),
                other => return Err(PbzError::new_err(format!("unsupported dtype {other:?}"))),
            };
            let total = (sum_len) * bed_sources.len() as u64 * elem as u64;
            config.progress = Some(progress::make_sink(&track, total));
        }

        match dtype.as_str() {
            "int32" => {
                rs_from_bed::<i32>(&mut store, &track, &bed_sources, column_idx, genome, config)
            }
            "float32" => {
                rs_from_bed::<f32>(&mut store, &track, &bed_sources, column_idx, genome, config)
            }
            "bool" => {
                rs_from_bed::<bool>(&mut store, &track, &bed_sources, column_idx, genome, config)
            }
            other => return Err(PbzError::new_err(format!("unsupported dtype {other:?}"))),
        }
        .map_err(|e| PbzError::new_err(format!("{e}")))?;
        Ok(())
    })
}

/// Single-pass, multi-column import of one tabix-indexed BED into N scalar
/// tracks. `columns` is an ordered list of `(header_name, dtype)`; each becomes
/// a track named after the column. `genome` is a .fai / chrom.sizes path.
#[pyfunction]
#[pyo3(signature = (store_path, bed_gz, columns, genome, workers=None, chunk_size=None, shard_size=None, progress=false))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bed_multi(
    py: Python<'_>,
    store_path: String,
    bed_gz: String,
    columns: Vec<(String, String)>,
    genome: String,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    shard_size: Option<usize>,
    progress: bool,
) -> PyResult<()> {
    py.allow_threads(|| {
        if columns.is_empty() {
            return Err(PbzError::new_err("bed multi import: no columns"));
        }
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let genome = Genome::from_fai(&genome).map_err(|e| PbzError::new_err(format!("{e}")))?;

        let parsed: Vec<(String, Dtype)> = columns
            .iter()
            .map(|(name, ds)| {
                Dtype::from_str(ds)
                    .map(|dt| (name.clone(), dt))
                    .map_err(|e| PbzError::new_err(format!("column {name:?}: {e}")))
            })
            .collect::<PyResult<_>>()?;
        let schema = BedSchema(
            parsed
                .iter()
                .map(|(name, dt)| BedColumnSpec::named(name.clone(), *dt))
                .collect(),
        );

        let mut config = Config::default();
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(s) = shard_size {
            config.shard_size = Some(s);
        }
        if progress {
            let sum_len: u64 = genome.contigs().iter().map(|c| c.length).sum();
            let per_pos: u64 = parsed.iter().map(|(_, dt)| dtype_bytes(*dt)).sum();
            config.progress = Some(progress::make_sink("bed-multi", sum_len * per_pos));
        }

        rs_from_bed_multi(&mut store, Path::new(&bed_gz), &schema, genome, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))?;
        Ok(())
    })
}

/// Create a new empty flat pbz store (a bare `zarr_conventions` marker root).
/// The imports open this store and add tracks to it.
#[pyfunction]
fn create_store(store_path: String) -> PyResult<()> {
    PbzStore::create(&store_path)
        .map(|_| ())
        .map_err(|e| PbzError::new_err(format!("{e}")))
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
    m.add_function(wrap_pyfunction!(import_bed, m)?)?;
    m.add_function(wrap_pyfunction!(import_bed_multi, m)?)?;
    m.add_function(wrap_pyfunction!(create_store, m)?)?;
    m.add_function(wrap_pyfunction!(d4_contigs, m)?)?;
    m.add_function(wrap_pyfunction!(bigwig_contigs, m)?)?;
    Ok(())
}
