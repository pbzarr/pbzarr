//! PyO3 bindings for pbzarr: `import_d4`, `import_bigwig`, `import_bed`, and
//! `import_bam`.

use std::path::PathBuf;

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pbzarr::ExplicitArraySpec;
use pbzarr::Genome;
use pbzarr::PbzStore;
use pbzarr::import::progress::make_sink;
use pbzarr::import::{Config, Report, Source};
use pbzarr::io::Dtype;
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, ColumnSelector, DepthFilter, ImportMode, InferRows,
    OverlapMode, column_index_by_name, execute_bed_schema_plan, from_bam as rs_from_bam,
    from_bed_matrix as rs_from_bed_matrix, from_bigwig as rs_from_bigwig, from_d4 as rs_from_d4,
    infer_bed_dtypes_for_sources, plan_bed_schema, read_bed_layout,
};

create_exception!(_native, PbzError, PyRuntimeError);

/// Parse the optional `codecs` JSON kwarg into `config.codecs`, rejecting
/// the chunk/shard kwargs it replaces.
fn apply_codecs_kwarg(
    config: &mut Config,
    codecs: Option<&str>,
    conflicts: &[(&str, bool)],
) -> PyResult<()> {
    let Some(raw) = codecs else {
        return Ok(());
    };
    if let Some((name, _)) = conflicts.iter().find(|(_, present)| *present) {
        return Err(PbzError::new_err(format!(
            "codecs conflicts with {name}; the codecs spec controls chunking and sharding"
        )));
    }
    let spec = ExplicitArraySpec::parse(raw).map_err(|e| PbzError::new_err(format!("{e}")))?;
    config.codecs = Some(spec);
    Ok(())
}

/// Bulk-import one or more d4 files into a new `int32` track.
///
/// The track is created from the source headers; it must NOT already exist.
/// d4 stores depths as `int32` natively, so import is zero-conversion.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, column_dim=None, workers=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, scales=None, in_flight=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_d4(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    column_dim: Option<String>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    scales: Option<Vec<u64>>,
    in_flight: Option<usize>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let report = py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let d4_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim,
            ..Config::default()
        };
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(n) = in_flight {
            config.in_flight_spans = n;
        }
        if let Some(n) = decode_chunks {
            config.decode_chunks = n;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(c) = column_chunk_size {
            config.column_chunk_size = Some(c);
        }
        if let Some(s) = shard_size {
            config.shard_size = Some(s);
        }
        if let Some(s) = shard_column_size {
            config.shard_column_size = Some(s);
        }
        if progress {
            config.progress = Some(make_sink(&track));
        }
        config.scales = scales.unwrap_or_default();
        apply_codecs_kwarg(
            &mut config,
            codecs.as_deref(),
            &[
                ("chunk_size", chunk_size.is_some()),
                ("column_chunk_size", column_chunk_size.is_some()),
                ("shard_size", shard_size.is_some()),
                ("shard_column_size", shard_column_size.is_some()),
            ],
        )?;
        rs_from_d4(&mut store, &track, &d4_sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))
    })?;
    report_to_dict(py, &report)
}

/// Bulk-import one or more bigWig files into a new `float32` track.
///
/// The track is created from the source headers; it must NOT already exist.
/// bigWig stores values as `float32` natively. A bigWig holds no value at an
/// uncovered base, so uncovered positions read back as `NaN` and reductions
/// skip them.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, column_dim=None, workers=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, scales=None, in_flight=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bigwig(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    column_dim: Option<String>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    scales: Option<Vec<u64>>,
    in_flight: Option<usize>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let report = py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let bw_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim,
            ..Config::default()
        };
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(n) = in_flight {
            config.in_flight_spans = n;
        }
        if let Some(n) = decode_chunks {
            config.decode_chunks = n;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(c) = column_chunk_size {
            config.column_chunk_size = Some(c);
        }
        if let Some(s) = shard_size {
            config.shard_size = Some(s);
        }
        if let Some(s) = shard_column_size {
            config.shard_column_size = Some(s);
        }
        if progress {
            config.progress = Some(make_sink(&track));
        }
        config.scales = scales.unwrap_or_default();
        apply_codecs_kwarg(
            &mut config,
            codecs.as_deref(),
            &[
                ("chunk_size", chunk_size.is_some()),
                ("column_chunk_size", column_chunk_size.is_some()),
                ("shard_size", shard_size.is_some()),
                ("shard_column_size", shard_column_size.is_some()),
            ],
        )?;
        rs_from_bigwig(&mut store, &track, &bw_sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))
    })?;
    report_to_dict(py, &report)
}

/// Build a `BedSchema` from `(selector, dtype)` entries. An all-digit selector
/// is a 1-based BED column number; anything else is a header name. A numeric
/// entry's track takes the header name at that column, or `field{N}` for
/// headerless input. `track_override` (the single-column form) renames the one
/// output track. `None` dtypes are inferred across every source.
fn build_bed_schema(
    sources: &[Source],
    entries: &[(String, Option<String>)],
    track_override: Option<&str>,
) -> PyResult<BedSchema> {
    let first = sources
        .first()
        .ok_or_else(|| PbzError::new_err("bed import: no sources"))?;
    let mut header: Option<Option<Vec<String>>> = None; // first source's header, read once
    let mut specs = Vec::with_capacity(entries.len());
    let mut file_cols = Vec::with_capacity(entries.len());
    for (selector, dtype) in entries {
        let is_index = !selector.is_empty() && selector.bytes().all(|b| b.is_ascii_digit());
        let (selector, file_col, default_track) = if is_index {
            let number: usize = selector.parse().map_err(|_| {
                PbzError::new_err(format!(
                    "bed import: numeric BED-column selector {selector:?} is out of range"
                ))
            })?;
            let Some(index) = number.checked_sub(1) else {
                return Err(PbzError::new_err(
                    "bed import: BED column selectors are 1-based; 0 selects nothing",
                ));
            };
            if header.is_none() {
                header = Some(
                    read_bed_layout(&first.path)
                        .map_err(|e| PbzError::new_err(format!("{e}")))?
                        .header,
                );
            }
            let track = header
                .as_ref()
                .and_then(|names| names.as_ref())
                .and_then(|names| names.get(index))
                .cloned()
                .unwrap_or_else(|| format!("field{number}"));
            (ColumnSelector::Index(index), index, track)
        } else {
            let index = column_index_by_name(&first.path, selector)
                .map_err(|e| PbzError::new_err(format!("{e}")))?;
            (
                ColumnSelector::Name(selector.clone()),
                index,
                selector.clone(),
            )
        };
        file_cols.push(file_col);
        let dtype = dtype
            .as_deref()
            .map(Dtype::from_str)
            .transpose()
            .map_err(|e| PbzError::new_err(format!("{e}")))?;
        specs.push((selector, dtype, default_track));
    }

    let missing: Vec<usize> = specs
        .iter()
        .zip(&file_cols)
        .filter(|((_, dtype, _), _)| dtype.is_none())
        .map(|(_, file_col)| *file_col)
        .collect();
    let mut inferred = if missing.is_empty() {
        Vec::new()
    } else {
        infer_bed_dtypes_for_sources(sources, &missing, InferRows::Sample(1_000))
            .map_err(|e| PbzError::new_err(format!("{e}")))?
    }
    .into_iter();

    let columns = specs
        .into_iter()
        .map(|(selector, dtype, default_track)| {
            let dtype = match dtype {
                Some(dtype) => dtype,
                None => inferred.next().ok_or_else(|| {
                    PbzError::new_err("bed import: dtype inference returned too few dtypes")
                })?,
            };
            Ok(BedColumnSpec {
                selector,
                dtype,
                track_name: Some(track_override.map(str::to_owned).unwrap_or(default_track)),
                description: None,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(BedSchema(columns))
}

/// Import bgzipped, tabix-indexed BED sources into new tracks. `genome` is a
/// .fai / chrom.sizes path (BED files carry no contig lengths).
///
/// Three forms, mirroring `pbz import bed`:
/// - `column` (a header name, or an all-digit 1-based BED column number)
///   imports that one BED column into one track, named by `track` when given.
///   `dtype` overrides inference.
/// - `schema` entries `(selector, dtype)` import one track per entry; a `None`
///   dtype is inferred across every source.
/// - Neither: BED3 imports a bool presence track and headerless BED4 one value
///   track (both need `track`); headered input imports every value column,
///   as one wide track when a single source has `track` set, else as one
///   track per column.
///
/// Any 2D output (several sources, or a wide track) requires `column_dim`.
/// Returns the import report dict.
#[pyfunction]
#[pyo3(signature = (store_path, sources, genome, schema=None, track=None, column=None, dtype=None, column_dim=None, workers=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, scales=None, in_flight=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bed(
    py: Python<'_>,
    store_path: String,
    sources: Vec<(String, Option<String>)>,
    genome: String,
    schema: Option<Vec<(String, Option<String>)>>,
    track: Option<String>,
    column: Option<String>,
    dtype: Option<String>,
    column_dim: Option<String>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    scales: Option<Vec<u64>>,
    in_flight: Option<usize>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let report = py.allow_threads(|| {
        let bed_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        if bed_sources.is_empty() {
            return Err(PbzError::new_err("bed import: no sources"));
        }
        let genome = Genome::from_fai(&genome).map_err(|e| PbzError::new_err(format!("{e}")))?;

        let mut config = Config {
            column_dim,
            ..Config::default()
        };
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(n) = in_flight {
            config.in_flight_spans = n;
        }
        if let Some(n) = decode_chunks {
            config.decode_chunks = n;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(c) = column_chunk_size {
            config.column_chunk_size = Some(c);
        }
        if let Some(s) = shard_size {
            config.shard_size = Some(s);
        }
        if let Some(s) = shard_column_size {
            config.shard_column_size = Some(s);
        }
        if progress {
            config.progress = Some(make_sink(track.as_deref().unwrap_or("bed")));
        }
        config.scales = scales.unwrap_or_default();
        apply_codecs_kwarg(
            &mut config,
            codecs.as_deref(),
            &[
                ("chunk_size", chunk_size.is_some()),
                ("column_chunk_size", column_chunk_size.is_some()),
                ("shard_size", shard_size.is_some()),
                ("shard_column_size", shard_column_size.is_some()),
            ],
        )?;
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;

        let entries = match (schema, &column) {
            (Some(_), Some(_)) => {
                return Err(PbzError::new_err(
                    "bed import: schema conflicts with track, column, and dtype",
                ));
            }
            (Some(entries), None) => {
                if track.is_some() || dtype.is_some() {
                    return Err(PbzError::new_err(
                        "bed import: schema conflicts with track, column, and dtype",
                    ));
                }
                Some(entries)
            }
            (None, Some(column)) => Some(vec![(column.clone(), dtype.clone())]),
            (None, None) => None,
        };
        match entries {
            Some(entries) => {
                let track_override = if column.is_some() {
                    track.as_deref()
                } else {
                    None
                };
                let bed_schema = build_bed_schema(&bed_sources, &entries, track_override)?;
                let plan = plan_bed_schema(&bed_sources, &bed_schema, &config)
                    .map_err(|e| PbzError::new_err(format!("{e}")))?;
                execute_bed_schema_plan(&mut store, plan, genome, config)
                    .map_err(|e| PbzError::new_err(format!("{e}")))
            }
            None => {
                let options = BedImportOptions {
                    track: track.clone(),
                    dtype: dtype
                        .as_deref()
                        .map(Dtype::from_str)
                        .transpose()
                        .map_err(|e| PbzError::new_err(format!("{e}")))?,
                    ..BedImportOptions::default()
                };
                rs_from_bed_matrix(&mut store, &bed_sources, genome, &options, config)
                    .map_err(|e| PbzError::new_err(format!("{e}")))
            }
        }
    })?;
    report_to_dict(py, &report)
}

/// Turn a `Report` into the `dict` handed back to Python callers.
fn report_to_dict(py: Python<'_>, report: &Report) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("contigs_written", report.contigs_written)?;
    dict.set_item("bytes_written", report.bytes_written)?;
    dict.set_item("tasks_completed", report.tasks_completed)?;
    dict.set_item("tasks_skipped", report.tasks_skipped)?;
    dict.set_item("pieces", report.pieces)?;
    dict.set_item("wall_seconds", report.wall_seconds)?;
    dict.set_item("decode_seconds", report.decode_seconds)?;
    dict.set_item("probe_seconds", report.probe_seconds)?;
    dict.set_item("write_seconds", report.write_seconds)?;
    dict.set_item("worker_busy_seconds", report.worker_busy_seconds)?;
    dict.set_item("worker_idle_seconds", report.worker_idle_seconds)?;
    dict.set_item("gate_wait_seconds", report.gate_wait_seconds)?;
    dict.set_item("handle_wait_seconds", report.handle_wait_seconds)?;
    Ok(dict.into_any().unbind())
}

/// Bulk-import per-base depth or composition counts from one or more
/// BAM/CRAM files into a new `int32` track (or tracks, for composition mode).
///
/// `mode` is `"depth"` (single `track` output) or `"composition"` (`track`
/// plus one `track_{field}` per composition field, per `from_bam`'s track
/// naming). CRAM sources need `reference`; BAM sources ignore it. `overlap`
/// is `"proper"` (default, mosdepth-matched: only PROPER_PAIR-flagged
/// overlapping mates are deduped), `"all"` (riker/samtools-mpileup-style
/// unconditional dedup), or `"none"` (dedup disabled).
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, mode="depth".to_string(), reference=None, min_mapq=0, exclude_flags=1796, min_bq=0, overlap="proper".to_string(), count_deletions=false, column_dim=None, workers=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, scales=None, in_flight=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bam(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    mode: String,
    reference: Option<String>,
    min_mapq: u8,
    exclude_flags: u16,
    min_bq: u8,
    overlap: String,
    count_deletions: bool,
    column_dim: Option<String>,
    workers: Option<usize>,
    chunk_size: Option<usize>,
    column_chunk_size: Option<usize>,
    shard_size: Option<usize>,
    shard_column_size: Option<usize>,
    progress: bool,
    codecs: Option<String>,
    scales: Option<Vec<u64>>,
    in_flight: Option<usize>,
    decode_chunks: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let mode = match mode.as_str() {
        "depth" => ImportMode::Depth,
        "composition" => ImportMode::Composition,
        other => {
            return Err(PbzError::new_err(format!(
                "unsupported bam import mode {other:?} (expected \"depth\" or \"composition\")"
            )));
        }
    };
    let overlap = match overlap.as_str() {
        "proper" => OverlapMode::ProperOnly,
        "all" => OverlapMode::All,
        "none" => OverlapMode::None,
        other => {
            return Err(PbzError::new_err(format!(
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
    let report = py.allow_threads(|| {
        let mut store =
            PbzStore::open(&store_path).map_err(|e| PbzError::new_err(format!("{e}")))?;
        let bam_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim,
            ..Config::default()
        };
        if let Some(w) = workers {
            config.workers = w;
        }
        if let Some(n) = in_flight {
            config.in_flight_spans = n;
        }
        if let Some(n) = decode_chunks {
            config.decode_chunks = n;
        }
        if let Some(c) = chunk_size {
            config.chunk_size = Some(c);
        }
        if let Some(c) = column_chunk_size {
            config.column_chunk_size = Some(c);
        }
        if let Some(s) = shard_size {
            config.shard_size = Some(s);
        }
        if let Some(s) = shard_column_size {
            config.shard_column_size = Some(s);
        }
        if progress {
            config.progress = Some(make_sink(&track));
        }
        config.scales = scales.unwrap_or_default();
        apply_codecs_kwarg(
            &mut config,
            codecs.as_deref(),
            &[
                ("chunk_size", chunk_size.is_some()),
                ("column_chunk_size", column_chunk_size.is_some()),
                ("shard_size", shard_size.is_some()),
                ("shard_column_size", shard_column_size.is_some()),
            ],
        )?;
        rs_from_bam(
            &mut store,
            &track,
            &bam_sources,
            mode,
            filter,
            reference.map(PathBuf::from),
            config,
        )
        .map_err(|e| PbzError::new_err(format!("{e}")))
    })?;
    report_to_dict(py, &report)
}

/// Create a new empty flat pbz store (a bare `zarr_conventions` marker root).
/// The imports open this store and add tracks to it.
#[pyfunction]
fn create_store(store_path: String) -> PyResult<()> {
    PbzStore::create(&store_path)
        .map(|_| ())
        .map_err(|e| PbzError::new_err(format!("{e}")))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PbzError", m.py().get_type::<PbzError>())?;
    m.add_function(wrap_pyfunction!(import_d4, m)?)?;
    m.add_function(wrap_pyfunction!(import_bigwig, m)?)?;
    m.add_function(wrap_pyfunction!(import_bed, m)?)?;
    m.add_function(wrap_pyfunction!(import_bam, m)?)?;
    m.add_function(wrap_pyfunction!(create_store, m)?)?;
    Ok(())
}
