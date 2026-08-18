use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, debug, info, warn};
use pbzarr::import::Config;
use pbzarr::import::progress::{self, make_sink};
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore};
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, ColumnSelector, DepthFilter, ImportMode, InferRows,
    OverlapMode, column_index_by_name, execute_bed_schema_plan, from_bam, from_bed_matrix,
    infer_bed_dtypes, infer_bed_dtypes_for_sources, plan_bed_schema, read_bed_layout,
};

#[derive(Debug, Parser)]
#[command(name = "pbz", version, about = "Per-base Zarr tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect a PBZ store.
    View(ViewArgs),
    /// Import data into a PBZ store.
    Import(Box<ImportArgs>),
}

#[derive(Debug, Args)]
struct ViewArgs {
    #[command(flatten)]
    global: GlobalOptions,
    /// PBZ store to inspect.
    store: PathBuf,
    /// Track to view (defaults to first track if multiple present)
    track: Option<String>,
}

#[derive(Debug, Args)]
struct ImportArgs {
    #[command(subcommand)]
    format: ImportCommand,
}

#[derive(Debug, Subcommand)]
enum ImportCommand {
    /// Import from tabix-indexed BED files.
    Bed(BedArgs),
    /// Import per-base depth or composition counts from BAM/CRAM files.
    Bam(BamArgs),
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bed -o cohort.pbz --genome genome.fai --track depth -c sample s1.bed.gz s2.bed.gz\n  pbz import bed -o mask.pbz --genome genome.fai --track callable regions.bed.gz\n  pbz import bed -o stats.pbz --genome genome.fai -c metric --schema schema.tsv wide.bed.gz"
)]
struct BedArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    bed: BedOptions,
}

#[derive(Debug, Args)]
struct GlobalOptions {
    /// Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings
    /// and errors always print. `RUST_LOG` overrides this when set.
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

impl GlobalOptions {
    fn log_level(&self) -> LevelFilter {
        match self.verbose {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            2 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    }
}

/// Install the logger at the verbosity `-v` asked for. `RUST_LOG` wins when
/// set, so a user can dial in one noisy module without flooding the rest.
///
/// The logger is bridged through the progress module's `MultiProgress`, which
/// suspends the bars while a record prints; otherwise log lines and the
/// animated bar would overwrite each other on stderr.
fn init_logging(global: &GlobalOptions) {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(global.log_level());
    if let Ok(spec) = std::env::var("RUST_LOG") {
        builder.parse_filters(&spec);
    }
    let logger = builder.build();
    // `try_init` does not set the max level itself, so read it off the logger
    // first (per indicatif-log-bridge's documented pattern).
    let level = logger.filter();
    if LogWrapper::new(progress::multi().clone(), logger)
        .try_init()
        .is_ok()
    {
        log::set_max_level(level);
    }
}

#[derive(Debug, Args)]
struct ImportOptions {
    /// Number of import workers.
    #[arg(short = 't', long, default_value_t = 4)]
    threads: usize,
    /// Position chunk size.
    #[arg(long)]
    chunk_size: Option<usize>,
    /// Column chunk size for multi-source imports.
    #[arg(long)]
    column_chunk_size: Option<usize>,
    /// Position shard size.
    #[arg(long)]
    shard_size: Option<usize>,
    /// Column shard size for multi-source imports.
    #[arg(long)]
    shard_column_size: Option<usize>,
    /// Name for the column dimension of a 2D track (required whenever the
    /// output is 2D: several sources, or several BED columns in one track).
    #[arg(short = 'c', long)]
    column_dim: Option<String>,
    /// Show import progress on stderr. This is the default; the flag is
    /// accepted so scripts can be explicit.
    #[arg(long)]
    progress: bool,
    /// Hide the import progress display.
    #[arg(long, conflicts_with = "progress")]
    no_progress: bool,
}

impl ImportOptions {
    /// Base pipeline config, with a progress sink labeled `label` unless
    /// `--no-progress` turned it off. The pipeline sizes the sink itself, so
    /// the label is all it needs here.
    fn config(&self, label: &str) -> Config {
        Config {
            workers: self.threads,
            chunk_size: self.chunk_size,
            column_chunk_size: self.column_chunk_size,
            shard_size: self.shard_size,
            shard_column_size: self.shard_column_size,
            column_dim: self.column_dim.clone(),
            progress: (!self.no_progress).then(|| make_sink(label)),
        }
    }
}

#[derive(Debug, Args)]
#[group(required = true, multiple = false)]
struct ImportInputOptions {
    /// Input files, optionally suffixed with `:LABEL`.
    #[arg(value_name = "PATH[:LABEL]")]
    input_files: Vec<Source>,
    /// Tab-delimited source manifest containing PATH or PATH<TAB>LABEL records.
    #[arg(short, long, value_name = "PATH")]
    file_list: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Source {
    path: PathBuf,
    label: Option<String>,
}

impl Source {
    fn validate_labels(sources: &[Self]) -> Result<()> {
        let has_labels = sources.iter().any(|source| source.label.is_some());
        let has_unlabeled = sources.iter().any(|source| source.label.is_none());
        if has_labels && has_unlabeled {
            bail!("sources must either all provide labels or none do");
        }
        Ok(())
    }
}

impl FromStr for Source {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (path, label) = match value.rsplit_once(':') {
            Some((path, label)) => (path, Some(label)),
            None => (value, None),
        };
        if path.is_empty() {
            return Err("source path must not be empty".into());
        }
        if label == Some("") {
            return Err("source label must not be empty".into());
        }
        Ok(Self {
            path: PathBuf::from(path),
            label: label.map(str::to_owned),
        })
    }
}

fn read_source_list(path: &Path) -> Result<Vec<Source>> {
    let file = File::open(path).with_context(|| format!("open source list {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut sources = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read source list line {}", index + 1))?;
        let record = line.trim();
        if record.is_empty() || record.starts_with('#') {
            continue;
        }
        let fields = record.split('\t').collect::<Vec<_>>();
        if fields.len() > 2 || fields[0].is_empty() {
            bail!(
                "source list {} line {} must be PATH or PATH<TAB>LABEL",
                path.display(),
                index + 1
            );
        }
        let label = fields.get(1).map(|label| label.to_string());
        if label.as_deref() == Some("") {
            bail!(
                "source list {} line {} has an empty label",
                path.display(),
                index + 1
            );
        }
        sources.push(Source {
            path: PathBuf::from(fields[0]),
            label,
        });
    }

    if sources.is_empty() {
        bail!("source list {} contains no BED sources", path.display());
    }
    Source::validate_labels(&sources)?;
    Ok(sources)
}

impl ImportInputOptions {
    fn resolve_input(&self) -> Result<Vec<Source>> {
        let sources = match &self.file_list {
            Some(path) => read_source_list(path)?,
            None => self.input_files.clone(),
        };
        if sources.is_empty() {
            bail!("at least one input source is required");
        }
        Source::validate_labels(&sources)?;
        Ok(sources)
    }
}

#[derive(Debug, Args)]
struct BedOptions {
    /// PBZ output store. It is created when absent.
    #[arg(short('o'), long, required_unless_present = "emit_schema")]
    output: Option<PathBuf>,
    #[command(flatten)]
    input: ImportInputOptions,
    /// FAI or chromosome-sizes file defining the genome.
    #[arg(long, required_unless_present = "emit_schema")]
    genome: Option<PathBuf>,
    /// BED header column to import (requires a `#`-prefixed header). Repeating
    /// selects several BED columns; omit to import all BED value columns.
    #[arg(long = "field", conflicts_with = "schema")]
    fields: Vec<String>,
    /// Output track name. Required for BED3, headerless BED4, and when several
    /// BED columns from one source form one wide track.
    #[arg(long, conflicts_with = "schema")]
    track: Option<String>,
    /// Value dtype for headerless BED4 input (inferred when omitted).
    #[arg(long, conflicts_with = "schema")]
    dtype: Option<String>,
    /// TSV schema mapping BED columns to target tracks. Unique target names
    /// create one track per BED column; repeated one-target rows create a
    /// single BED-column-axis track. A BED column is a header name or 1-based
    /// column index.
    #[arg(long, value_name = "PATH")]
    schema: Option<PathBuf>,
    /// Print a schema TSV for one BED source (BED column, target track,
    /// inferred dtype) to stdout and exit without importing. Edit it and feed
    /// it back via --schema.
    #[arg(long, conflicts_with_all = ["schema", "fields", "track", "dtype"])]
    emit_schema: bool,
    /// Records sampled for dtype inference, or `all`. A sample infers
    /// conservative classes (bool/int32/float32); `all` (or a file smaller
    /// than the sample) scans exhaustively and min-widths (uint8, int16, ...).
    #[arg(long, default_value = "1000", value_name = "N|all")]
    infer_rows: String,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bam -o depth.pbz --track depth s1.bam s2.bam\n  pbz import bam -o comp.pbz --track depth --mode composition --reference ref.fa sample.cram"
)]
struct BamArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    bam: BamOptions,
}

#[derive(Debug, Args)]
struct BamOptions {
    /// PBZ output store. It is created when absent.
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Output track name. In composition mode the depth track keeps this
    /// name and the per-event tracks are named `{track}_{field}`.
    #[arg(long)]
    track: String,
    /// Depth counts coverage only; composition also counts per-base events,
    /// emitting `{track}_{field}` for a, c, g, t, n, ins, del, ref_skip.
    #[arg(long, value_enum, default_value_t = ImportModeArg::Depth)]
    mode: ImportModeArg,
    /// Reference FASTA for CRAM sources (ignored for BAM).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Minimum mapping quality for a read to count.
    #[arg(long, default_value_t = 0)]
    min_mapq: u8,
    /// Minimum per-base quality for a base to count.
    #[arg(long, default_value_t = 0)]
    min_bq: u8,
    /// SAM flag bits that exclude a read (default: UNMAP|SECONDARY|QCFAIL|DUP).
    #[arg(long, default_value_t = 1796)]
    exclude_flags: u16,
    /// Mate-overlap dedup mode. `proper` (default) matches mosdepth: only
    /// PROPER_PAIR-flagged overlapping mates are collapsed to one count.
    /// `all` matches riker/samtools-mpileup-style unconditional dedup of
    /// any overlapping mate pair. `none` disables dedup, double-counting
    /// every overlapping pair's shared span.
    #[arg(long, value_enum, default_value_t = OverlapModeArg::Proper)]
    overlap: OverlapModeArg,
    /// Count CIGAR D (deletion)-spanned positions as covered depth. Off by
    /// default, matching the samtools/mosdepth/Picard/riker convention of
    /// excluding deletions from depth.
    #[arg(long)]
    count_deletions: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImportModeArg {
    Depth,
    Composition,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OverlapModeArg {
    Proper,
    All,
    None,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::View(args) => view(args),
        Command::Import(import) => match import.format {
            ImportCommand::Bed(args) => import_bed(args),
            ImportCommand::Bam(args) => import_bam(args),
        },
    }
}

fn view(args: ViewArgs) -> Result<()> {
    init_logging(&args.global);
    bail!("view is not implemented yet for {}", args.store.display())
}

fn import_bed(args: BedArgs) -> Result<()> {
    init_logging(&args.global);
    let sources: Vec<pbzarr::import::Source> = args
        .bed
        .input
        .resolve_input()?
        .into_iter()
        .map(|source| pbzarr::import::Source {
            path: source.path,
            column_label: source.label,
        })
        .collect();
    let infer_rows = parse_infer_rows(&args.bed.infer_rows)?;
    debug!(
        "resolved {} BED source(s), infer rows {infer_rows:?}",
        sources.len()
    );

    if args.bed.emit_schema {
        info!("emitting a schema scaffold to stdout; no store is written");
        if sources.len() != 1 {
            bail!("--emit-schema requires exactly one BED source");
        }
        return emit_schema(&sources[0].path, infer_rows);
    }

    let genome_path = args.bed.genome.context("--genome is required to import")?;
    let genome = Genome::from_fai(&genome_path)
        .with_context(|| format!("read genome {}", genome_path.display()))?;
    info!(
        "genome {}: {} contig(s)",
        genome_path.display(),
        genome.contigs().len()
    );
    let output = args.bed.output.context("--output is required to import")?;
    let label = args
        .bed
        .track
        .clone()
        .or_else(|| args.bed.schema.as_ref().map(|_| "bed-schema".to_owned()))
        .unwrap_or_else(|| "bed".to_owned());
    let config = args.import.config(&label);
    if let Some(schema_path) = &args.bed.schema {
        info!("loading BED schema {}", schema_path.display());
        let schema = load_schema(schema_path, &sources, infer_rows)?;
        let plan = plan_bed_schema(&sources, &schema, &config).context("plan BED schema")?;
        debug!("BED schema planned against {} source(s)", sources.len());
        let mut store = open_or_create(&output)?;
        execute_bed_schema_plan(&mut store, plan, genome, config).context("import BED schema")?;
    } else {
        let mut store = open_or_create(&output)?;
        let options = BedImportOptions {
            fields: (!args.bed.fields.is_empty()).then_some(args.bed.fields),
            track: args.bed.track,
            dtype: args
                .bed
                .dtype
                .as_deref()
                .map(Dtype::from_str)
                .transpose()
                .context("invalid --dtype")?,
            infer_rows,
            ..BedImportOptions::default()
        };
        from_bed_matrix(&mut store, &sources, genome, &options, config)
            .context("import BED columns")?;
    }
    info!("wrote {}", output.display());
    Ok(())
}

/// Open `path` as a store, creating it when it does not exist yet. Importing
/// into an existing store adds tracks to it, which is easy to do by accident
/// with a stale path, so say which of the two happened.
fn open_or_create(path: &Path) -> Result<PbzStore> {
    if path.exists() {
        warn!(
            "{} exists; adding track(s) to the existing store",
            path.display()
        );
        PbzStore::open(path).with_context(|| format!("open output store {}", path.display()))
    } else {
        debug!("creating store {}", path.display());
        PbzStore::create(path).with_context(|| format!("create output store {}", path.display()))
    }
}

fn import_bam(args: BamArgs) -> Result<()> {
    init_logging(&args.global);
    let sources: Vec<pbzarr::import::Source> = args
        .bam
        .input
        .resolve_input()?
        .into_iter()
        .map(|source| pbzarr::import::Source {
            path: source.path,
            column_label: source.label,
        })
        .collect();
    let n_sources = sources.len();
    debug!("resolved {n_sources} alignment source(s)");

    let config = args.import.config(&args.bam.track);
    let mode = match args.bam.mode {
        ImportModeArg::Depth => ImportMode::Depth,
        ImportModeArg::Composition => ImportMode::Composition,
    };
    let overlap = match args.bam.overlap {
        OverlapModeArg::Proper => OverlapMode::ProperOnly,
        OverlapModeArg::All => OverlapMode::All,
        OverlapModeArg::None => OverlapMode::None,
    };
    let filter = DepthFilter {
        min_mapq: args.bam.min_mapq,
        exclude_flags: args.bam.exclude_flags,
        min_bq: args.bam.min_bq,
        overlap,
        count_deletions: args.bam.count_deletions,
    };

    let output = args.bam.output;
    let mut store = open_or_create(&output)?;

    let report = from_bam(
        &mut store,
        &args.bam.track,
        &sources,
        mode,
        filter,
        args.bam.reference,
        config,
    )
    .context("import BAM/CRAM")?;

    debug!(
        "bam import wrote {} bytes across {} contig(s)",
        report.bytes_written, report.contigs_written
    );
    println!(
        "imported {n_sources} sources -> {} ({} tasks, {} skipped)",
        output.display(),
        report.tasks_completed,
        report.tasks_skipped
    );
    Ok(())
}

struct SchemaRow {
    bed_column: String,
    track: Option<String>,
    dtype: Option<Dtype>,
}

fn read_schema_rows(text: &str, path: &Path) -> Result<Vec<SchemaRow>> {
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let record = line.trim_end();
        if record.trim().is_empty() || record.trim_start().starts_with('#') {
            continue;
        }
        let cells = record.split('\t').collect::<Vec<_>>();
        if cells.len() > 3 || cells[0].is_empty() {
            bail!(
                "schema {} line {} must be BED_COLUMN[<TAB>TARGET_TRACK[<TAB>DTYPE]]",
                path.display(),
                index + 1
            );
        }
        let dtype = cells
            .get(2)
            .filter(|dtype| !dtype.is_empty())
            .map(|dtype| Dtype::from_str(dtype))
            .transpose()
            .with_context(|| format!("schema {} line {}", path.display(), index + 1))?;
        rows.push(SchemaRow {
            bed_column: cells[0].to_owned(),
            track: cells
                .get(1)
                .filter(|t| !t.is_empty())
                .map(|t| (*t).to_owned()),
            dtype,
        });
    }
    if rows.is_empty() {
        bail!("schema {} selects no BED columns", path.display());
    }
    Ok(rows)
}

/// Print a schema TSV scaffold for `bed` to stdout: one row per BED value
/// column mapped to a target track, with its inferred dtype. Unique target
/// names create per-BED-column tracks; repeating one target creates a single
/// BED-column-axis track. Headerless files use 1-based column selectors.
fn emit_schema(bed: &Path, infer_rows: InferRows) -> Result<()> {
    let layout =
        read_bed_layout(bed).with_context(|| format!("read layout of {}", bed.display()))?;
    if layout.n_cols == 3 {
        bail!(
            "{} is BED3; it has no BED value columns (import it directly with --track)",
            bed.display()
        );
    }
    let columns: Vec<usize> = (3..layout.n_cols).collect();
    let dtypes = infer_bed_dtypes(bed, &columns, infer_rows)
        .with_context(|| format!("infer dtypes for {}", bed.display()))?;
    println!("# BED column\ttarget track\tdtype");
    match &layout.header {
        Some(names) => {
            for (name, dtype) in names[3..].iter().zip(&dtypes) {
                println!("{name}\t{name}\t{}", dtype.as_str());
            }
        }
        None => {
            for (column, dtype) in columns.iter().zip(&dtypes) {
                println!("{}\tfield{}\t{}", column + 1, column + 1, dtype.as_str());
            }
        }
    }
    Ok(())
}

/// Build a `BedSchema` from a TSV schema file, resolving name selectors against
/// the first BED header and inferring omitted dtypes across every source.
fn load_schema(
    schema_path: &Path,
    sources: &[pbzarr::import::Source],
    infer_rows: InferRows,
) -> Result<BedSchema> {
    let text = std::fs::read_to_string(schema_path)
        .with_context(|| format!("read schema {}", schema_path.display()))?;
    let rows = read_schema_rows(&text, schema_path)?;
    let first = sources
        .first()
        .context("schema import requires a BED source")?;

    let mut specs = Vec::with_capacity(rows.len());
    let mut file_cols = Vec::with_capacity(rows.len());
    for row in &rows {
        let (selector, file_col) = if row.bed_column.bytes().all(|byte| byte.is_ascii_digit()) {
            let index = row.bed_column.parse::<usize>().map_err(|_| {
                anyhow!(
                    "invalid numeric BED-column selector {:?}: value overflows usize",
                    row.bed_column
                )
            })?;
            if index < 4 {
                bail!(
                    "schema BED column {index} is a coordinate column (indices are 1-based; value BED columns start at 4)"
                );
            }
            if row.track.is_none() {
                bail!(
                    "schema BED column {index} needs a target track name (no header name is available as a default)"
                );
            }
            (ColumnSelector::Index(index - 1), index - 1)
        } else {
            let file_col = column_index_by_name(&first.path, &row.bed_column)
                .with_context(|| format!("resolve schema BED column {:?}", row.bed_column))?;
            (ColumnSelector::Name(row.bed_column.clone()), file_col)
        };
        file_cols.push(file_col);
        specs.push(BedColumnSpec {
            selector,
            dtype: row.dtype.unwrap_or(Dtype::Bool),
            track_name: row.track.clone(),
            description: None,
        });
    }

    let missing: Vec<usize> = rows
        .iter()
        .zip(&file_cols)
        .filter(|(row, _)| row.dtype.is_none())
        .map(|(_, col)| *col)
        .collect();
    if !missing.is_empty() {
        let inferred = infer_bed_dtypes_for_sources(sources, &missing, infer_rows)
            .context("infer schema BED column dtypes across input sources")?;
        let mut inferred = inferred.into_iter();
        for (spec, row) in specs.iter_mut().zip(&rows) {
            if row.dtype.is_none() {
                spec.dtype = inferred
                    .next()
                    .context("inference returned fewer dtypes than requested")?;
            }
        }
    }
    Ok(BedSchema(specs))
}

fn parse_infer_rows(value: &str) -> Result<InferRows> {
    if value == "all" {
        return Ok(InferRows::All);
    }
    let rows = value.parse::<usize>().with_context(|| {
        format!("invalid --infer-rows value {value:?}; expected a positive integer or all")
    })?;
    if rows == 0 {
        bail!("--infer-rows must be positive or all");
    }
    Ok(InferRows::Sample(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_rows_parse_selectors_tracks_and_dtypes() {
        let text = "# BED column\ttarget track\tdtype\n4\tdepth\tint32\nmapq\n\ncallable\tcallable_mask\tbool\n";
        let rows = read_schema_rows(text, Path::new("schema.tsv")).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].bed_column, "4");
        assert_eq!(rows[0].track.as_deref(), Some("depth"));
        assert_eq!(rows[0].dtype, Some(Dtype::I32));
        assert_eq!(rows[1].bed_column, "mapq");
        assert!(rows[1].track.is_none());
        assert!(rows[1].dtype.is_none());
        assert_eq!(rows[2].track.as_deref(), Some("callable_mask"));
        assert_eq!(rows[2].dtype, Some(Dtype::Bool));
    }

    #[test]
    fn schema_rows_reject_wide_and_empty_records() {
        assert!(read_schema_rows("a\tb\tint32\td\n", Path::new("s.tsv")).is_err());
        assert!(read_schema_rows("# only comments\n", Path::new("s.tsv")).is_err());
    }
}
