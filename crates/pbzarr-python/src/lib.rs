//! PyO3 bindings for pbzarr: the imports, `create_dataset`, and `parse_region`.

use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pbzarr::Contig;
use pbzarr::ExplicitArraySpec;
use pbzarr::import::progress::line_sink;
use pbzarr::import::{Config, ConfigBuilder, ProgressSink, Report, Source, store_reference};
use pbzarr::io::{Dtype, ReaderError};
use pbzarr::{Dataset, PbzError as RsError, Reference, RegionQuery};
use pbzarr_readers::{
    BedColumnRequest, BedImportOptions, DepthFilter, ImportMode, InferRows, OverlapMode,
    build_bed_schema, execute_bed_schema_plan, from_bam as rs_from_bam,
    from_bed_matrix as rs_from_bed_matrix, from_bigwig as rs_from_bigwig, from_d4 as rs_from_d4,
    plan_bed_schema,
};

create_exception!(_native, PbzError, PyRuntimeError);

/// A missing file raises `FileNotFoundError`, a bad argument `ValueError`,
/// and every other failure `PbzError`.
fn py_err(error: RsError) -> PyErr {
    let missing = match &error {
        RsError::Io(source) | RsError::Reader(ReaderError::Io { source, .. }) => {
            source.kind() == ErrorKind::NotFound
        }
        _ => false,
    };
    let message = error.to_string();
    if missing {
        return PyFileNotFoundError::new_err(message);
    }
    match error {
        RsError::InvalidArgument(_)
        | RsError::InvalidDtype { .. }
        | RsError::InvalidRegion { .. }
        | RsError::ContigNotFound { .. } => PyValueError::new_err(message),
        _ => PbzError::new_err(message),
    }
}

/// Build the import `Config` from the keyword arguments every import binding
/// takes. A `codecs` JSON spec replaces the chunk and shard sizes, so the two
/// together fail. Progress goes to Python `sys.stderr`.
fn build_config(mut builder: ConfigBuilder, codecs: Option<&str>) -> PyResult<Config> {
    let progress = builder.progress.take();
    if let Some(raw) = codecs {
        if let Some(name) = builder.grid_size_set() {
            return Err(PyValueError::new_err(format!(
                "codecs conflicts with {name}; the codecs spec controls chunking and sharding"
            )));
        }
        let spec =
            ExplicitArraySpec::parse(raw).map_err(|e| PyValueError::new_err(e.to_string()))?;
        builder.codecs = Some(spec);
    }
    let mut config = builder.build();
    config.progress = progress.as_deref().map(python_progress);
    Ok(config)
}

/// Write progress lines through Python `sys.stderr`, so that Jupyter shows
/// them in the cell output.
fn python_progress(label: &str) -> Arc<dyn ProgressSink> {
    line_sink(label, |line| {
        Python::with_gil(|py| {
            let write = || -> PyResult<()> {
                let stderr = py.import("sys")?.getattr("stderr")?;
                stderr.call_method1("write", (format!("{line}\n"),))?;
                stderr.call_method0("flush")?;
                Ok(())
            };
            if let Err(error) = write() {
                error.print(py);
            }
        })
    })
}

const SIGNAL_CHECK: Duration = Duration::from_millis(100);

/// Run an import on another thread and check for Python signals while it
/// runs. On a signal such as Ctrl-C, cancel the import, wait for it to stop,
/// and raise the exception of the signal handler.
fn run_interruptible<T: Send>(
    py: Python<'_>,
    mut config: Config,
    import: impl FnOnce(Config) -> PyResult<T> + Send,
) -> PyResult<T> {
    let cancel = Arc::new(AtomicBool::new(false));
    config.cancel = Some(Arc::clone(&cancel));
    let (tx, mut rx) = channel();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = tx.send(import(config));
        });
        loop {
            let (back, received) = py.allow_threads(move || {
                let received = rx.recv_timeout(SIGNAL_CHECK);
                (rx, received)
            });
            rx = back;
            match received {
                Ok(result) => return result,
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(PbzError::new_err(
                        "the import thread stopped with no result",
                    ));
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            if let Err(signal) = py.check_signals() {
                cancel.store(true, Ordering::Relaxed);
                let _ = py.allow_threads(move || rx.recv());
                return Err(signal);
            }
        }
    })
}

fn to_sources(sources: Vec<(PathBuf, Option<String>)>) -> PyResult<Vec<Source>> {
    if sources.is_empty() {
        return Err(PyValueError::new_err("sources must not be empty"));
    }
    let sources = sources
        .into_iter()
        .map(|(path, column_label)| Source { path, column_label })
        .collect::<Vec<_>>();
    Source::check_labels(&sources).map_err(py_err)?;
    Ok(sources)
}

fn parse_dtype(dtype: Option<&str>) -> PyResult<Option<Dtype>> {
    dtype.map(Dtype::from_str).transpose().map_err(py_err)
}

/// Bulk-import one or more d4 files into a new `int32` track.
///
/// The store is created from the source headers when the path does not
/// exist. The track must not already be in the store. d4 stores depths as
/// `int32` natively, so import is zero-conversion.
#[pyfunction]
#[pyo3(signature = (*, destination, track, sources, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_d4(
    py: Python<'_>,
    destination: PathBuf,
    track: String,
    sources: Vec<(PathBuf, Option<String>)>,
    dim: Option<String>,
    threads: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let sources = to_sources(sources)?;
    let config = build_config(
        ConfigBuilder {
            workers: threads,
            decode_chunks,
            chunk_size,
            column_chunk_size,
            shard_size,
            shard_column_size,
            column_dim: dim,
            codecs: None,
            progress: progress.then(|| track.clone()),
        },
        codecs.as_deref(),
    )?;
    let report = run_interruptible(py, config, |config| {
        rs_from_d4(destination.as_path(), &track, &sources, config).map_err(py_err)
    })?;
    report_to_dict(py, &report)
}

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// The store is created from the source headers when the path does not
/// exist. The track must not already be in the store. bigWig stores values
/// as `float32` natively. A bigWig holds no value at an uncovered base, so
/// uncovered positions read back as `NaN` and reductions skip them.
#[pyfunction]
#[pyo3(signature = (*, destination, track, sources, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bigwig(
    py: Python<'_>,
    destination: PathBuf,
    track: String,
    sources: Vec<(PathBuf, Option<String>)>,
    dim: Option<String>,
    threads: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let sources = to_sources(sources)?;
    let config = build_config(
        ConfigBuilder {
            workers: threads,
            decode_chunks,
            chunk_size,
            column_chunk_size,
            shard_size,
            shard_column_size,
            column_dim: dim,
            codecs: None,
            progress: progress.then(|| track.clone()),
        },
        codecs.as_deref(),
    )?;
    let report = run_interruptible(py, config, |config| {
        rs_from_bigwig(destination.as_path(), &track, &sources, None, config).map_err(py_err)
    })?;
    report_to_dict(py, &report)
}

/// Import bgzipped, tabix-indexed BED sources into new tracks. `genome` is a
/// .fai / chrom.sizes path (BED files carry no contig lengths). It is needed
/// only to create the store; an existing store gives its own lengths.
///
/// Three forms, mirroring `pbz import bed`:
/// - `column` (a header name, or a 1-based BED column number) imports that
///   one BED column into one track. `track` names it, and a numbered column
///   needs `track`. `dtype` overrides inference.
/// - `schema` entries `(column, dtype)` import one track per entry, named by
///   the column; a `None` dtype is inferred across every source.
/// - Neither: BED3 imports a bool presence track and headerless BED4 one value
///   track (both need `track`); headered input imports every value column,
///   as one wide track when a single source has `track` set, else as one
///   track per column.
///
/// Any 2D output (several sources, or a wide track) requires `dim`.
/// Returns the import report dict.
#[pyfunction]
#[pyo3(signature = (*, destination, sources, genome=None, schema=None, track=None, column=None, dtype=None, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bed(
    py: Python<'_>,
    destination: PathBuf,
    sources: Vec<(PathBuf, Option<String>)>,
    genome: Option<PathBuf>,
    schema: Option<Vec<(String, Option<String>)>>,
    track: Option<String>,
    column: Option<String>,
    dtype: Option<String>,
    dim: Option<String>,
    threads: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let sources = to_sources(sources)?;
    let dtype = parse_dtype(dtype.as_deref())?;
    let requests = match (schema, column) {
        (Some(_), column) if column.is_some() || track.is_some() || dtype.is_some() => {
            return Err(PyValueError::new_err(
                "schema conflicts with track, column, and dtype",
            ));
        }
        (Some(entries), _) => Some(
            entries
                .into_iter()
                .map(|(column, dtype)| {
                    Ok(BedColumnRequest {
                        column,
                        track: None,
                        dtype: parse_dtype(dtype.as_deref())?,
                        fill: None,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        ),
        (None, Some(column)) => Some(vec![BedColumnRequest {
            column,
            track: track.clone(),
            dtype,
            fill: None,
        }]),
        (None, None) => None,
    };
    let config = build_config(
        ConfigBuilder {
            workers: threads,
            decode_chunks,
            chunk_size,
            column_chunk_size,
            shard_size,
            shard_column_size,
            column_dim: dim,
            codecs: None,
            progress: progress.then(|| track.as_deref().unwrap_or("bed").to_owned()),
        },
        codecs.as_deref(),
    )?;
    let report = run_interruptible(py, config, |config| {
        let genome = genome
            .map(Reference::from_fai)
            .transpose()
            .and_then(|genome| store_reference(&destination, genome))
            .map_err(py_err)?;
        match requests {
            Some(requests) => {
                let schema = build_bed_schema(&sources, &requests, None, InferRows::default())
                    .map_err(py_err)?;
                let plan = plan_bed_schema(&sources, &schema, &config).map_err(py_err)?;
                execute_bed_schema_plan(destination.as_path(), plan, genome, config).map_err(py_err)
            }
            None => {
                let options = BedImportOptions {
                    track,
                    dtype,
                    ..BedImportOptions::default()
                };
                rs_from_bed_matrix(destination.as_path(), &sources, genome, &options, config)
                    .map_err(py_err)
            }
        }
    })?;
    report_to_dict(py, &report)
}

/// Turn a `Report` into the `dict` handed back to Python callers.
fn report_to_dict(py: Python<'_>, report: &Report) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("positions", report.positions)?;
    dict.set_item("chunks_written", report.chunks_written)?;
    dict.set_item("chunks_skipped", report.chunks_skipped)?;
    dict.set_item("bytes_written", report.bytes_written)?;
    dict.set_item("wall_seconds", report.wall_seconds)?;
    Ok(dict.into_any().unbind())
}

/// Bulk-import per-base depth or composition counts from one or more
/// BAM/CRAM files into a new `int32` track (or tracks, for composition mode).
///
/// `mode` is `"depth"` (single `track` output) or `"composition"` (`track`
/// plus one `track_{field}` per composition field, per `from_bam`'s track
/// naming). CRAM sources need `fasta`; BAM sources ignore it. `overlap`
/// is `"proper"` (default, mosdepth-matched: only PROPER_PAIR-flagged
/// overlapping mates are deduped), `"all"` (riker/samtools-mpileup-style
/// unconditional dedup), or `"none"` (dedup disabled).
#[pyfunction]
#[pyo3(signature = (*, destination, track, sources, mode="depth".to_string(), fasta=None, min_mapq=0, exclude_flags=1796, min_bq=0, overlap="proper".to_string(), count_deletions=false, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bam(
    py: Python<'_>,
    destination: PathBuf,
    track: String,
    sources: Vec<(PathBuf, Option<String>)>,
    mode: String,
    fasta: Option<PathBuf>,
    min_mapq: u8,
    exclude_flags: u16,
    min_bq: u8,
    overlap: String,
    count_deletions: bool,
    dim: Option<String>,
    threads: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mode = match mode.as_str() {
        "depth" => ImportMode::Depth,
        "composition" => ImportMode::Composition,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported bam import mode {other:?} (expected \"depth\" or \"composition\")"
            )));
        }
    };
    let overlap = match overlap.as_str() {
        "proper" => OverlapMode::ProperOnly,
        "all" => OverlapMode::All,
        "none" => OverlapMode::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported overlap mode {other:?} (expected \"proper\", \"all\", or \"none\")"
            )));
        }
    };
    let filter = DepthFilter {
        min_mapq,
        exclude_flags,
        min_bq,
        overlap,
        count_deletions,
    };
    let sources = to_sources(sources)?;
    let config = build_config(
        ConfigBuilder {
            workers: threads,
            decode_chunks,
            chunk_size,
            column_chunk_size,
            shard_size,
            shard_column_size,
            column_dim: dim,
            codecs: None,
            progress: progress.then(|| track.clone()),
        },
        codecs.as_deref(),
    )?;
    let report = run_interruptible(py, config, |config| {
        rs_from_bam(
            destination.as_path(),
            &track,
            &sources,
            mode,
            filter,
            fasta,
            config,
        )
        .map_err(py_err)
    })?;
    report_to_dict(py, &report)
}

/// Create a new empty format-0.5 Dataset store for `pbz.write`.
///
/// The root Dataset's `reference`, ordered contig name/length pairs, is fixed
/// here and shared by every track the store later holds. Import does not need
/// this: it creates the store from the source headers.
#[pyfunction]
#[pyo3(signature = (*, destination, reference))]
fn create_dataset(destination: PathBuf, reference: Vec<(String, u64)>) -> PyResult<()> {
    let contigs = reference
        .into_iter()
        .map(|(name, length)| Contig { name, length })
        .collect();
    Reference::new(contigs)
        .and_then(|reference| Dataset::create(&destination, reference))
        .map(|_| ())
        .map_err(py_err)
}

/// Parse a region string into `(contig, start, end)`, 0-based and half-open,
/// with `None` for an open start or end. When `contigs` contains the whole
/// string, the string is that contig and is not split on `':'`.
#[pyfunction]
#[pyo3(signature = (*, text, contigs=None))]
fn parse_region(
    text: &str,
    contigs: Option<&Bound<'_, PyAny>>,
) -> PyResult<(String, Option<u64>, Option<u64>)> {
    let mut lookup_error = None;
    let parsed = RegionQuery::parse_with(text, |name| match contigs {
        Some(contigs) => contigs.contains(name).unwrap_or_else(|e| {
            lookup_error = Some(e);
            false
        }),
        None => false,
    });
    if let Some(e) = lookup_error {
        return Err(e);
    }
    let query = parsed.map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok((query.contig, query.start, query.end))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PbzError", m.py().get_type::<PbzError>())?;
    m.add_function(wrap_pyfunction!(import_d4, m)?)?;
    m.add_function(wrap_pyfunction!(import_bigwig, m)?)?;
    m.add_function(wrap_pyfunction!(import_bed, m)?)?;
    m.add_function(wrap_pyfunction!(import_bam, m)?)?;
    m.add_function(wrap_pyfunction!(create_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(parse_region, m)?)?;
    Ok(())
}
