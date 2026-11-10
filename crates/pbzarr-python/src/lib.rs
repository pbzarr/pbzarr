//! PyO3 bindings for pbzarr: `import_d4`, `import_bigwig`, `import_bed`, and
//! `import_bam`.

use std::path::{Path, PathBuf};

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use pbzarr::Contig;
use pbzarr::ExplicitArraySpec;
use pbzarr::import::progress::make_sink;
use pbzarr::import::{Config, Report, Source, store_reference};
use pbzarr::io::Dtype;
use pbzarr::{Dataset, Reference};
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
/// The store is created from the source headers when the path does not
/// exist. The track must not already be in the store. d4 stores depths as
/// `int32` natively, so import is zero-conversion.
#[pyfunction]
#[pyo3(signature = (store_path, track, sources, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_d4(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
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
    let report = py.allow_threads(|| {
        let d4_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim: dim,
            ..Config::default()
        };
        if let Some(n) = threads {
            config.workers = n;
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
        rs_from_d4(Path::new(&store_path), &track, &d4_sources, config)
            .map_err(|e| PbzError::new_err(format!("{e}")))
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
#[pyo3(signature = (store_path, track, sources, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bigwig(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
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
    let report = py.allow_threads(|| {
        let bw_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim: dim,
            ..Config::default()
        };
        if let Some(n) = threads {
            config.workers = n;
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
        rs_from_bigwig(Path::new(&store_path), &track, &bw_sources, config)
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
/// .fai / chrom.sizes path (BED files carry no contig lengths). It is needed
/// only to create the store; an existing store gives its own lengths.
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
/// Any 2D output (several sources, or a wide track) requires `dim`.
/// Returns the import report dict.
#[pyfunction]
#[pyo3(signature = (store_path, sources, genome=None, schema=None, track=None, column=None, dtype=None, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bed(
    py: Python<'_>,
    store_path: String,
    sources: Vec<(String, Option<String>)>,
    genome: Option<String>,
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
        let genome = genome
            .map(Reference::from_fai)
            .transpose()
            .and_then(|genome| store_reference(Path::new(&store_path), genome))
            .map_err(|e| PbzError::new_err(format!("{e}")))?;

        let mut config = Config {
            column_dim: dim,
            ..Config::default()
        };
        if let Some(n) = threads {
            config.workers = n;
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
                execute_bed_schema_plan(Path::new(&store_path), plan, genome, config)
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
                rs_from_bed_matrix(
                    Path::new(&store_path),
                    &bed_sources,
                    genome,
                    &options,
                    config,
                )
                .map_err(|e| PbzError::new_err(format!("{e}")))
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
#[pyo3(signature = (store_path, track, sources, mode="depth".to_string(), fasta=None, min_mapq=0, exclude_flags=1796, min_bq=0, overlap="proper".to_string(), count_deletions=false, dim=None, threads=None, chunk_size=None, column_chunk_size=None, shard_size=None, shard_column_size=None, progress=false, codecs=None, decode_chunks=None))]
#[allow(clippy::too_many_arguments)] // signature mirrors the Python keyword API
fn import_bam(
    py: Python<'_>,
    store_path: String,
    track: String,
    sources: Vec<(String, Option<String>)>,
    mode: String,
    fasta: Option<String>,
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
        let bam_sources: Vec<Source> = sources
            .iter()
            .map(|(path, column_label)| Source {
                path: path.into(),
                column_label: column_label.clone(),
            })
            .collect();
        let mut config = Config {
            column_dim: dim,
            ..Config::default()
        };
        if let Some(n) = threads {
            config.workers = n;
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
            Path::new(&store_path),
            &track,
            &bam_sources,
            mode,
            filter,
            fasta.map(PathBuf::from),
            config,
        )
        .map_err(|e| PbzError::new_err(format!("{e}")))
    })?;
    report_to_dict(py, &report)
}

/// Create a new empty format-0.5 pbz store for `pbz.write`.
///
/// The root Dataset's `reference`, ordered contig name/length pairs, is fixed
/// here and shared by every track the store later holds. Import does not need
/// this: it creates the store from the source headers.
#[pyfunction]
#[pyo3(signature = (store_path, reference))]
fn create_store(store_path: String, reference: Vec<(String, u64)>) -> PyResult<()> {
    let contigs = reference
        .into_iter()
        .map(|(name, length)| Contig { name, length })
        .collect();
    Reference::new(contigs)
        .and_then(|reference| Dataset::create(&store_path, reference))
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
