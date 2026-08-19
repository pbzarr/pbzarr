//! Schema-driven, single-pass import of tabix-indexed BED columns. Unique
//! target track names create one track per BED column. Mapping every selected
//! column to one target creates a rank-2 track with an explicitly labeled,
//! homogeneous BED-column axis. PBZ permits only one non-position axis, so a
//! grouped BED-column axis cannot be combined with a multi-source axis.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use log::{debug, info};
use noodles_bgzf as bgzf;
use noodles_csi::BinningIndex;

use pbzarr::genome::Genome;
use pbzarr::import::{Config, Import, PipelineOptions, Report, Source};
use pbzarr::io::{
    ColumnSinkMut, Dtype, OutputField, OutputSchema, OutputSinkMut, ReaderError, ValueReader,
};
use pbzarr::{
    PbzError, PbzStore, Result, TrackConfig, validate_column_dimension_name, validate_node_name,
};

use crate::coords::noodles_query_interval;

use super::schema::open_bgzf;
use super::schema::read_bed_layout;
use super::schema::{BedImportOptions, ResolvedImport, resolve_sources};

/// Reader-side result alias (parse/IO errors), distinct from `pbzarr::Result`.
type IoResult<T> = std::result::Result<T, ReaderError>;

/// Zero fill matching the dtype's zero so all-gap chunks are elided on write.
fn zero_fill(dtype: Dtype) -> serde_json::Value {
    match dtype {
        Dtype::F32 | Dtype::F64 => serde_json::json!(0.0),
        Dtype::Bool => serde_json::json!(false),
        _ => serde_json::json!(0),
    }
}

/// How a schema entry picks a BED value column.
pub enum ColumnSelector {
    /// Absolute 0-based file column index (`>= 3`).
    Index(usize),
    /// A name from the `#`-prefixed header line.
    Name(String),
}

/// One column of a [`BedSchema`]: which file column, its dtype, and the target
/// track. `track_name` defaults to the header name for `Name` selectors and is
/// required for `Index` selectors. Uncovered positions always take the dtype's
/// zero fill; a non-zero fill knob is deliberately not offered (gaps and true
/// zeros are indistinguishable).
pub struct BedColumnSpec {
    pub selector: ColumnSelector,
    pub dtype: Dtype,
    pub track_name: Option<String>,
    pub description: Option<String>,
}

impl BedColumnSpec {
    /// Select a column by header name; the track takes the same name.
    pub fn named(name: impl Into<String>, dtype: Dtype) -> Self {
        Self {
            selector: ColumnSelector::Name(name.into()),
            dtype,
            track_name: None,
            description: None,
        }
    }

    /// Select a column by absolute file index; a track name is required.
    pub fn indexed(index: usize, dtype: Dtype, track_name: impl Into<String>) -> Self {
        Self {
            selector: ColumnSelector::Index(index),
            dtype,
            track_name: Some(track_name.into()),
            description: None,
        }
    }
}

/// An ordered mapping from BED columns to target tracks.
///
/// [`plan_bed_schema`] gives columns with unique target names separate tracks.
/// If every selected column has the same target name, it instead creates one
/// rank-2 track whose explicitly labeled BED-column axis is homogeneous: all
/// selected columns must share a dtype and description. Because PBZ supports
/// only one non-position axis, that grouped layout accepts one BED source.
pub struct BedSchema(pub Vec<BedColumnSpec>);

/// A schema entry with its selector resolved to a concrete file column + name.
pub(super) struct Resolved {
    pub file_col: usize,
    pub column_label: String,
    pub dtype: Dtype,
    pub track_name: String,
    pub description: Option<String>,
}

/// A fully resolved and validated schema import. Construct this before
/// opening or creating the output store, then pass it to
/// [`execute_bed_schema_plan`]. Its internals are deliberately private so an
/// invalid pipeline shape cannot be assembled by callers.
pub struct BedSchemaPlan {
    sources: Vec<pbzarr::import::Source>,
    layout: SchemaLayout,
    column_dim: Option<String>,
}

enum SchemaLayout {
    PerColumn(Vec<Resolved>),
    BedColumnAxis {
        track_name: String,
        dtype: Dtype,
        columns: Vec<usize>,
        column_labels: Vec<String>,
        description: Option<String>,
    },
}

/// Resolve and validate a BED schema against all input sources without
/// touching an output store.
pub fn plan_bed_schema(
    sources: &[pbzarr::import::Source],
    schema: &BedSchema,
    config: &Config,
) -> Result<BedSchemaPlan> {
    let first = sources
        .first()
        .ok_or_else(|| PbzError::Metadata("bed schema import: no input sources".into()))?;
    if schema.0.is_empty() {
        return Err(PbzError::Metadata(
            "bed schema import: empty schema selects no BED columns".into(),
        ));
    }
    validate_column_dim(config.column_dim.as_deref())?;

    let layout = read_bed_layout(&first.path)?;
    for source in &sources[1..] {
        let other = read_bed_layout(&source.path)?;
        if other.n_cols != layout.n_cols {
            return Err(PbzError::Metadata(format!(
                "bed schema import: {} has {} BED columns but {} has {}",
                source.path.display(),
                other.n_cols,
                first.path.display(),
                layout.n_cols
            )));
        }
        if other.header != layout.header {
            return Err(PbzError::Metadata(format!(
                "bed schema import: {} BED column header differs from {}",
                source.path.display(),
                first.path.display()
            )));
        }
    }
    let mut resolved = Vec::with_capacity(schema.0.len());
    let mut selected_columns = HashSet::with_capacity(schema.0.len());
    for spec in &schema.0 {
        let (file_col, default_track, column_label, indexed_number) = match &spec.selector {
            ColumnSelector::Name(name) => {
                let Some(header) = layout.header.as_ref() else {
                    return Err(PbzError::Metadata(format!(
                        "bed schema import: named BED column {name:?} requires a #-prefixed header"
                    )));
                };
                let matching = header
                    .iter()
                    .enumerate()
                    .filter(|(_, column)| *column == name)
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>();
                let Some(&index) = matching.first() else {
                    return Err(PbzError::Metadata(format!(
                        "bed schema import: BED column {name:?} is not present in the header"
                    )));
                };
                if matching.len() > 1 {
                    return Err(PbzError::Metadata(format!(
                        "bed schema import: named BED column {name:?} is ambiguous because it matches {} header columns; use a one-based numeric selector",
                        matching.len()
                    )));
                }
                (index, Some(name.clone()), name.clone(), None)
            }
            ColumnSelector::Index(index) => {
                let one_based = indexed_column_number(*index)?;
                let label = match layout.header.as_ref().and_then(|header| header.get(*index)) {
                    Some(label) => label.clone(),
                    None => format!("column {one_based}"),
                };
                (*index, None, label, Some(one_based))
            }
        };
        if file_col < 3 {
            let selector = indexed_number
                .map(|number| format!("{number} (one-based)"))
                .unwrap_or_else(|| format!("{column_label:?}"));
            return Err(PbzError::Metadata(format!(
                "bed schema import: BED column {selector} is a coordinate column (value BED columns start at 4)"
            )));
        }
        if file_col >= layout.n_cols {
            let selector = indexed_number
                .map(|number| format!("{number} (one-based)"))
                .unwrap_or_else(|| format!("{column_label:?}"));
            return Err(PbzError::Metadata(format!(
                "bed schema import: BED column {selector} is outside the {}-column input",
                layout.n_cols
            )));
        }
        if !selected_columns.insert(file_col) {
            return Err(PbzError::Metadata(format!(
                "bed schema import: BED column {column_label:?} is selected more than once"
            )));
        }
        let track_name = spec.track_name.clone().or(default_track).ok_or_else(|| {
            let selector = indexed_number
                .map(|number| format!("{number} (one-based)"))
                .unwrap_or_else(|| format!("{column_label:?}"));
            PbzError::Metadata(format!(
                "bed schema import: BED column {selector} needs an explicit target track name"
            ))
        })?;
        validate_node_name(&track_name).map_err(|error| {
            PbzError::Metadata(format!(
                "bed schema import: target track name {track_name:?} is invalid: {error}"
            ))
        })?;
        resolved.push(Resolved {
            file_col,
            column_label,
            dtype: spec.dtype,
            track_name,
            description: spec.description.clone(),
        });
    }

    let unique_tracks = resolved
        .iter()
        .map(|column| column.track_name.as_str())
        .collect::<HashSet<_>>()
        .len();
    let layout = if unique_tracks == resolved.len() {
        if sources.len() > 1 && config.column_dim.is_none() {
            return Err(PbzError::Metadata(
                "bed schema import: multiple input sources make each BED column track rank 2 (position, source); set a column dimension name"
                    .into(),
            ));
        }
        if sources.len() > 1 {
            let mut source_labels = HashSet::with_capacity(sources.len());
            for label in sources.iter().map(Source::label) {
                if !source_labels.insert(label.clone()) {
                    return Err(PbzError::Metadata(format!(
                        "bed schema import: duplicate source-axis label {label:?}"
                    )));
                }
            }
        }
        SchemaLayout::PerColumn(resolved)
    } else if unique_tracks == 1 && resolved.len() >= 2 {
        let track_name = resolved[0].track_name.clone();
        if sources.len() > 1 {
            let bed_axis = config.column_dim.as_deref().unwrap_or("BED column");
            return Err(PbzError::Metadata(format!(
                "bed schema import: {} BED columns map to track {track_name:?}, creating BED-column axis {bed_axis:?}; {} sources create a source axis. Together they require rank 3 (position, source, {bed_axis}), but PBZ supports only one non-position axis. Use one input or give every BED column a unique track name",
                resolved.len(),
                sources.len()
            )));
        }
        if config.column_dim.is_none() {
            return Err(PbzError::Metadata(format!(
                "bed schema import: multiple BED columns map to track {track_name:?}, producing rank 2 (position, BED column); set a column dimension name"
            )));
        }
        let dtype = resolved[0].dtype;
        if resolved.iter().any(|column| column.dtype != dtype) {
            return Err(PbzError::Metadata(format!(
                "bed schema import: BED columns mapped to track {track_name:?} must share one dtype"
            )));
        }
        let description = resolved[0].description.clone();
        if resolved
            .iter()
            .any(|column| column.description.as_deref() != description.as_deref())
        {
            return Err(PbzError::Metadata(format!(
                "bed schema import: BED columns mapped to track {track_name:?} must have descriptions that match exactly, including whether a description is present"
            )));
        }
        let mut column_labels = HashSet::with_capacity(resolved.len());
        if let Some(duplicate) = resolved
            .iter()
            .map(|column| column.column_label.as_str())
            .find(|label| !column_labels.insert(*label))
        {
            return Err(PbzError::Metadata(format!(
                "bed schema import: grouped track {track_name:?} has duplicate BED column label {duplicate:?}"
            )));
        }
        SchemaLayout::BedColumnAxis {
            track_name,
            dtype,
            columns: resolved.iter().map(|column| column.file_col).collect(),
            column_labels: resolved
                .iter()
                .map(|column| column.column_label.clone())
                .collect(),
            description,
        }
    } else {
        return Err(PbzError::Metadata(
            "bed schema import: partially grouped BED columns are not supported; map all BED columns to one track or give every BED column a unique track name"
                .into(),
        ));
    };

    Ok(BedSchemaPlan {
        sources: sources.to_vec(),
        layout,
        column_dim: config.column_dim.clone(),
    })
}

fn indexed_column_number(index: usize) -> Result<usize> {
    index.checked_add(1).ok_or_else(|| {
        PbzError::Metadata(format!(
            "bed schema import: BED column index {index} cannot be represented as a one-based column number"
        ))
    })
}

fn validate_column_dim(column_dim: Option<&str>) -> Result<()> {
    let Some(name) = column_dim else {
        return Ok(());
    };
    validate_column_dimension_name(name)
        .map_err(|error| PbzError::Metadata(format!("bed schema import: {error}")))
}

/// Shared, thread-safe reader state: index + name→id map + resolved columns.
struct MultiShared {
    path: PathBuf,
    index: noodles_tabix::Index,
    ref_ids: HashMap<String, usize>,
    columns: Vec<(Option<usize>, Dtype, Option<String>)>,
    schema: OutputSchema,
    genome: Genome,
}

/// Multi-column BED reader. One tabix seek per region; each line's cells are
/// scattered into the matching per-track sink.
pub struct BedMultiReader {
    shared: Arc<MultiShared>,
    bgzf: bgzf::io::Reader<File>,
}

impl BedMultiReader {
    /// Open a bgzipped, tabix-indexed BED. `columns` is `(file column, dtype,
    /// column name)` in track order; every file column must be `>= 3`. A
    /// `None` column is a presence column: covered positions read as `1`
    /// (BED3 masks). The name, when one is known, becomes the output schema
    /// field name. `genome`'s contig names must match the BED's.
    pub fn open<P: AsRef<Path>>(
        bed_gz: P,
        columns: Vec<(Option<usize>, Dtype, Option<String>)>,
        genome: Genome,
    ) -> IoResult<Self> {
        let path = bed_gz.as_ref().to_path_buf();
        let tbi = PathBuf::from(format!("{}.tbi", path.display()));
        let index = noodles_tabix::fs::read(&tbi).map_err(|source| ReaderError::Io {
            path: tbi.clone(),
            source,
        })?;
        let header = index.header().ok_or_else(|| {
            ReaderError::Other(anyhow::anyhow!(
                "tabix index {} has no header",
                tbi.display()
            ))
        })?;
        let ref_ids = header
            .reference_sequence_names()
            .iter()
            .enumerate()
            .map(|(i, n)| (String::from_utf8_lossy(n.as_ref()).into_owned(), i))
            .collect();
        let schema = OutputSchema::new(
            columns
                .iter()
                .map(|(_, dtype, name)| OutputField {
                    name: name.clone(),
                    dtype: *dtype,
                })
                .collect(),
        );
        let bgzf = open_bgzf(&path)?;
        Ok(Self {
            shared: Arc::new(MultiShared {
                path,
                index,
                ref_ids,
                columns,
                schema,
                genome,
            }),
            bgzf,
        })
    }

    fn fork_handle(&self) -> IoResult<Self> {
        let bgzf = open_bgzf(&self.shared.path)?;
        Ok(Self {
            shared: Arc::clone(&self.shared),
            bgzf,
        })
    }
}

/// Parse one BED line and fill each target sink across the clipped run.
fn scatter_line(
    shared: &MultiShared,
    line: &[u8],
    contig_name: &str,
    win_start: u64,
    win_end: u64,
    sinks: &mut [ColumnSinkMut<'_>],
) -> IoResult<()> {
    let text = std::str::from_utf8(line).map_err(|e| {
        ReaderError::Other(anyhow::anyhow!(
            "non-utf8 line in {}: {e}",
            shared.path.display()
        ))
    })?;
    let mut fields = text.trim_end().split('\t');
    let chrom = fields.next().unwrap_or_default();
    if chrom != contig_name {
        return Ok(()); // tabix bins may overlap neighbors; skip foreign contigs
    }
    let cols: Vec<&str> = std::iter::once(chrom).chain(fields).collect();
    let parse_u64 = |s: &str| {
        s.parse::<u64>().map_err(|e| {
            ReaderError::Other(anyhow::anyhow!(
                "bad coord {s:?} in {}: {e}",
                shared.path.display()
            ))
        })
    };
    let bed_start = parse_u64(cols.get(1).copied().unwrap_or_default())?;
    let bed_end = parse_u64(cols.get(2).copied().unwrap_or_default())?;

    let lo = bed_start.max(win_start);
    let hi = bed_end.min(win_end);
    if lo >= hi {
        return Ok(());
    }
    let (rel_lo, rel_hi) = ((lo - win_start) as usize, (hi - win_start) as usize);

    for (col_idx, (file_col, _dt, _name)) in shared.columns.iter().enumerate() {
        let cell = match file_col {
            Some(file_col) => cols.get(*file_col).copied().ok_or_else(|| {
                ReaderError::Other(anyhow::anyhow!(
                    "line has no column {} in {}",
                    file_col,
                    shared.path.display()
                ))
            })?,
            None => "1",
        };
        sinks[col_idx].fill_run(rel_lo, rel_hi, cell)?;
    }
    Ok(())
}

/// One tabix seek per window; every overlapping line's cells are scattered
/// into the matching sink.
fn scatter_region(
    shared: &MultiShared,
    bgzf: &mut bgzf::io::Reader<File>,
    contig_name: &str,
    start: u64,
    end: u64,
    sinks: &mut [ColumnSinkMut<'_>],
) -> IoResult<()> {
    let Some(interval) = noodles_query_interval(start, end)? else {
        return Ok(());
    };
    let Some(&ref_id) = shared.ref_ids.get(contig_name) else {
        return Ok(()); // contig absent from this BED -> leave as caller's fill
    };

    let chunks = shared
        .index
        .query(ref_id, interval)
        .map_err(|source| ReaderError::Io {
            path: shared.path.clone(),
            source,
        })?;

    let mut line = Vec::new();
    for chunk in chunks {
        bgzf.seek(chunk.start()).map_err(|source| ReaderError::Io {
            path: shared.path.clone(),
            source,
        })?;
        while bgzf.virtual_position() < chunk.end() {
            line.clear();
            let n = bgzf
                .read_until(b'\n', &mut line)
                .map_err(|source| ReaderError::Io {
                    path: shared.path.clone(),
                    source,
                })?;
            if n == 0 {
                break;
            }
            if line.first() == Some(&b'#') {
                continue;
            }
            scatter_line(shared, &line, contig_name, start, end, sinks)?;
        }
    }
    Ok(())
}

/// Reborrow an output sink as a [`ColumnSinkMut`], whose `fill_run` parses raw
/// text cells into the sink's dtype.
fn column_sink<'a>(output: &'a mut OutputSinkMut<'_>) -> ColumnSinkMut<'a> {
    match output {
        OutputSinkMut::U8(v) => ColumnSinkMut::U8(v.view_mut()),
        OutputSinkMut::U16(v) => ColumnSinkMut::U16(v.view_mut()),
        OutputSinkMut::U32(v) => ColumnSinkMut::U32(v.view_mut()),
        OutputSinkMut::I8(v) => ColumnSinkMut::I8(v.view_mut()),
        OutputSinkMut::I16(v) => ColumnSinkMut::I16(v.view_mut()),
        OutputSinkMut::I32(v) => ColumnSinkMut::I32(v.view_mut()),
        OutputSinkMut::F32(v) => ColumnSinkMut::F32(v.view_mut()),
        OutputSinkMut::F64(v) => ColumnSinkMut::F64(v.view_mut()),
        OutputSinkMut::Bool(v) => ColumnSinkMut::Bool(v.view_mut()),
    }
}

impl ValueReader for BedMultiReader {
    fn contigs(&self) -> &Genome {
        &self.shared.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.shared.schema
    }

    fn read_into(
        &mut self,
        contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> IoResult<()> {
        if end <= start {
            return Ok(());
        }
        if outputs.len() != self.shared.schema.len() {
            return Err(ReaderError::SchemaMismatch {
                message: format!(
                    "bed reader produces {} field(s) but the engine handed {} sink(s)",
                    self.shared.schema.len(),
                    outputs.len()
                ),
            });
        }
        let window = (end - start) as usize;
        if let Some(output) = outputs.iter().find(|output| output.len() != window) {
            return Err(ReaderError::SchemaMismatch {
                message: format!(
                    "bed reader window holds {window} position(s) but a sink holds {}",
                    output.len()
                ),
            });
        }
        let Self { shared, bgzf } = self;
        let mut sinks: Vec<ColumnSinkMut<'_>> = outputs.iter_mut().map(column_sink).collect();
        scatter_region(shared, bgzf, contig, start, end, &mut sinks)
    }

    fn fork(&self) -> IoResult<Self> {
        self.fork_handle()
    }
}

/// Execute a plan returned by [`plan_bed_schema`]. Schema resolution is not
/// repeated. The executor rejects a column dimension that differs from the
/// planned value before it accesses the store.
pub fn execute_bed_schema_plan(
    store: &mut PbzStore,
    plan: BedSchemaPlan,
    genome: Genome,
    config: Config,
) -> Result<Report> {
    if config.column_dim != plan.column_dim {
        return Err(PbzError::Metadata(format!(
            "bed schema import: column dimension changed after planning (planned {:?}, execution requested {:?})",
            plan.column_dim, config.column_dim
        )));
    }
    info!(
        "bed schema import: {} source(s) into {}",
        plan.sources.len(),
        match &plan.layout {
            SchemaLayout::PerColumn(resolved) => format!("{} per-column track(s)", resolved.len()),
            SchemaLayout::BedColumnAxis {
                track_name,
                columns,
                ..
            } => format!("track {track_name:?} with {} BED column(s)", columns.len()),
        }
    );
    match plan.layout {
        SchemaLayout::PerColumn(resolved) => {
            let scalar = plan.sources.len() == 1 && config.column_dim.is_none();
            let source_labels = plan.sources.iter().map(Source::label).collect::<Vec<_>>();
            let specs = resolved
                .iter()
                .map(|column| {
                    let labels = (!scalar).then(|| source_labels.clone());
                    let mut track_config = config.track_config(
                        column.dtype,
                        Some(zero_fill(column.dtype)),
                        labels,
                        "sample",
                    );
                    if let Some(description) = &column.description {
                        track_config = track_config.description(description.clone());
                    }
                    (column.track_name.clone(), genome.clone(), track_config)
                })
                .collect();
            let reader_columns = resolved
                .iter()
                .map(|column| {
                    (
                        Some(column.file_col),
                        column.dtype,
                        Some(column.column_label.clone()),
                    )
                })
                .collect::<Vec<_>>();
            let readers = open_readers(&plan.sources, &reader_columns, &genome)?;
            let column_labels = (!scalar).then_some(source_labels);
            run_field_tracks(store, specs, readers, column_labels, &config)
        }
        SchemaLayout::BedColumnAxis {
            track_name,
            dtype,
            columns,
            column_labels,
            description,
        } => {
            let mut track_config = config.track_config(
                dtype,
                Some(zero_fill(dtype)),
                Some(column_labels.clone()),
                "BED column",
            );
            if let Some(description) = description {
                track_config = track_config.description(description);
            }
            let reader_columns = columns
                .into_iter()
                .zip(&column_labels)
                .map(|(column, label)| (Some(column), dtype, Some(label.clone())))
                .collect();
            let source = &plan.sources[0];
            let reader = BedMultiReader::open(&source.path, reader_columns, genome.clone())
                .map_err(|error| {
                    PbzError::Store(format!("open {}: {error}", source.path.display()))
                })?;
            let options = pipeline_options(&config);
            let names = vec![track_name.clone()];
            let report = store.create_tracks_with(
                vec![(track_name, genome, track_config)],
                move |tracks| {
                    Import::from_readers(vec![reader])?
                        .into_track(tracks[0])
                        .fields_as_columns()
                        .expect_column_labels(column_labels)
                        .options(options)
                        .run()
                },
            )?;
            config.scale_tracks(store, &names)?;
            Ok(report)
        }
    }
}

/// Import BED sources following the default shape table: BED3 becomes a bool
/// presence track, headerless BED4 a single value track, headered fields their
/// own tracks (2D over sources when there are several), and a single
/// multi-field source one wide 2D track whose columns are the fields. Any 2D
/// output requires `Config::column_dim`.
pub fn from_bed_matrix(
    store: &mut PbzStore,
    sources: &[pbzarr::import::Source],
    genome: Genome,
    options: &BedImportOptions,
    config: Config,
) -> Result<Report> {
    info!(
        "bed import: {} source(s), genome {} contig(s), {} positions",
        sources.len(),
        genome.contigs().len(),
        genome.contigs().iter().map(|c| c.length).sum::<u64>()
    );
    for source in sources {
        debug!("bed source {}", source.path.display());
    }
    let resolved = resolve_sources(sources, options)?;
    let two_d = matches!(resolved, ResolvedImport::Wide { .. }) || sources.len() > 1;
    if two_d && config.column_dim.is_none() {
        return Err(PbzError::Metadata(
            "bed import: 2D output requires a column dimension name".into(),
        ));
    }

    match resolved {
        ResolvedImport::SingleTrack {
            track,
            dtype,
            column,
            labels,
        } => {
            // Uncovered positions read back as zero; the on-disk fill must match
            // the zero-filled scratch buffers the pipeline writes, so all-gap
            // chunks elide.
            let scalar = sources.len() <= 1 && config.column_dim.is_none();
            let column_labels = (!scalar).then_some(labels);
            let cfg = config.track_config(
                dtype,
                Some(zero_fill(dtype)),
                column_labels.clone(),
                "sample",
            );
            let readers = open_readers(sources, &[(column, dtype, None)], &genome)?;
            run_field_tracks(
                store,
                vec![(track, genome, cfg)],
                readers,
                column_labels,
                &config,
            )
        }
        ResolvedImport::Wide {
            track,
            dtype,
            columns,
            column_labels,
        } => {
            // `two_d` above already requires `column_dim` whenever `Wide` is
            // resolved, so the column axis is unconditional here.
            let cfg = config.track_config(
                dtype,
                Some(zero_fill(dtype)),
                Some(column_labels.clone()),
                "column",
            );
            let reader_columns: Vec<(Option<usize>, Dtype, Option<String>)> = columns
                .into_iter()
                .zip(&column_labels)
                .map(|(c, label)| (Some(c), dtype, Some(label.clone())))
                .collect();
            let reader = BedMultiReader::open(&sources[0].path, reader_columns, genome.clone())
                .map_err(|e| PbzError::Store(format!("open {}: {e}", sources[0].path.display())))?;
            let options = pipeline_options(&config);
            let names = vec![track.clone()];
            let report = store.create_tracks_with(vec![(track, genome, cfg)], move |tracks| {
                Import::from_readers(vec![reader])?
                    .into_track(tracks[0])
                    .fields_as_columns()
                    .expect_column_labels(column_labels)
                    .options(options)
                    .run()
            })?;
            config.scale_tracks(store, &names)?;
            Ok(report)
        }
        ResolvedImport::PerField {
            fields,
            columns,
            labels,
        } => {
            let scalar = sources.len() <= 1 && config.column_dim.is_none();
            let specs = fields
                .iter()
                .map(|field| {
                    let column_labels = (!scalar).then(|| labels.clone());
                    let cfg = config.track_config(
                        field.dtype,
                        Some(zero_fill(field.dtype)),
                        column_labels,
                        "sample",
                    );
                    (field.name.clone(), genome.clone(), cfg)
                })
                .collect();
            let reader_columns: Vec<(Option<usize>, Dtype, Option<String>)> = columns
                .iter()
                .zip(&fields)
                .map(|(column, field)| (Some(*column), field.dtype, Some(field.name.clone())))
                .collect();
            let readers = open_readers(sources, &reader_columns, &genome)?;
            let column_labels = (!scalar).then_some(labels);
            run_field_tracks(store, specs, readers, column_labels, &config)
        }
    }
}

fn open_readers(
    sources: &[pbzarr::import::Source],
    columns: &[(Option<usize>, Dtype, Option<String>)],
    genome: &Genome,
) -> Result<Vec<BedMultiReader>> {
    sources
        .iter()
        .map(|source| {
            BedMultiReader::open(&source.path, columns.to_vec(), genome.clone())
                .map_err(|e| PbzError::Store(format!("open {}: {e}", source.path.display())))
        })
        .collect()
}

fn pipeline_options(config: &Config) -> PipelineOptions {
    PipelineOptions {
        workers: config.workers,
        progress: config.progress.clone(),
        ..PipelineOptions::default()
    }
}

fn run_field_tracks(
    store: &mut PbzStore,
    specs: Vec<(String, Genome, TrackConfig)>,
    readers: Vec<BedMultiReader>,
    // `Some` labels put the sources on the column axis; `None` keeps the
    // tracks rank-1. Source count alone cannot decide: one source with an
    // explicit `column_dim` is still rank 2, so the caller passes the same
    // labels it baked into the `TrackConfig`s.
    column_labels: Option<Vec<String>>,
    config: &Config,
) -> Result<Report> {
    let options = pipeline_options(config);
    let names: Vec<String> = specs.iter().map(|(name, _, _)| name.clone()).collect();
    let report = store.create_tracks_with(specs, move |tracks| {
        let mut builder = Import::from_readers(readers)?
            .into_tracks(tracks)
            .fields_as_tracks();
        if let Some(labels) = column_labels {
            builder = builder.readers_as_columns().expect_column_labels(labels);
        }
        builder.options(options).run()
    })?;
    config.scale_tracks(store, &names)?;
    Ok(report)
}
