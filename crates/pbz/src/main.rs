use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, debug, info, warn};
use pbzarr::import::progress::{self, make_sink};
use pbzarr::import::{
    Config, GenomeGeometry, LayoutEstimate, LayoutKnobs, TrackShape, auto_in_flight_spans,
};
use pbzarr::io::{Dtype, ValueReader as _};
use pbzarr::{ExplicitArraySpec, Genome, PbzStore, ScaleConfig};
use pbzarr_readers::{
    BamReader, BedColumnSpec, BedImportOptions, BedSchema, ColumnSelector, DepthFilter, ImportMode,
    InferRows, OverlapMode, column_index_by_name, execute_bed_schema_plan, from_bam,
    from_bed_matrix, infer_bed_dtypes, infer_bed_dtypes_for_sources, plan_bed_schema,
    read_bed_layout,
};

mod fmt;
mod stat;
mod view;

#[derive(Debug, Parser)]
#[command(name = "pbz", version, about = "Per-base Zarr tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Export one track as bedGraph-like text.
    View(ViewArgs),
    /// Import data into a PBZ store.
    Import(Box<ImportArgs>),
    /// Compute and publish a multiscale pyramid for a track.
    Scale(ScaleArgs),
}

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  pbz view cohort.pbz depth\n  pbz view -o depth.bedgraph.gz cohort.pbz depth\n  pbz view -c s1,s2 cohort.pbz af"
)]
struct ViewArgs {
    #[command(flatten)]
    global: GlobalOptions,
    /// PBZ store to export.
    store: PathBuf,
    /// Track to export. Optional when the store holds exactly one track.
    track: Option<String>,
    /// Column labels of a rank-2 track, comma-separated or repeated.
    /// Default: all columns in stored order.
    #[arg(short('c'), long, value_name = "LABEL", value_delimiter = ',')]
    columns: Vec<String>,
    /// Output path. A `.gz` suffix writes BGZF; default is stdout.
    #[arg(short('o'), long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Do not write the `#chrom start end ...` header line.
    #[arg(long)]
    no_header: bool,
    /// Worker threads for chunk decode and BGZF compression.
    /// Default: all cores.
    #[arg(short('t'), long, value_name = "N")]
    threads: Option<std::num::NonZeroUsize>,
    /// Significant digits for float values, `%g` style (UCSC bigWig
    /// convention).
    #[arg(short('p'), long, value_name = "N", default_value_t = 6)]
    precision: u8,
}

#[derive(Debug, Args)]
struct ImportArgs {
    #[command(subcommand)]
    format: ImportCommand,
}

#[derive(Debug, Args)]
struct ScaleArgs {
    #[command(flatten)]
    global: GlobalOptions,
    /// PBZ store containing the track.
    #[arg(short('z'), long, value_name = "PATH")]
    pbz: PathBuf,
    /// Track to build a pyramid for.
    #[arg(short('t'), long, value_name = "NAME")]
    track: String,
    /// Comma-separated downsampling factors (e.g. 8,64,512), strictly
    /// ascending with no duplicates. Defaults to the built-in ladder.
    #[arg(short('s'), long, value_name = "LIST")]
    scales: Option<String>,
    /// Number of scale workers.
    #[arg(short('j'), long, value_name = "N", default_value_t = 4)]
    workers: usize,
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
    after_help = "Examples:\n  pbz import bed -o cohort.pbz --genome genome.fai --track depth -c sample s1.bed.gz s2.bed.gz\n  pbz import bed -o mask.pbz --genome genome.fai --track callable regions.bed.gz\n  pbz import bed -o stats.pbz --genome genome.fai -c metric --schema schema.tsv wide.bed.gz\n  pbz import bed -o cohort.pbz --genome genome.fai --track depth -c sample --codecs '[{\"name\":\"transpose\",\"configuration\":{\"order\":[1,0]}},{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}]' s1.bed.gz s2.bed.gz"
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
    /// Position spans open at once in the import engine. Bounds the open
    /// chunk buffers, this many times --decode-chunks; more keeps workers
    /// busy near span boundaries. Default: auto (targets 3x the worker
    /// count in runnable pieces, clamped to 8..256).
    #[arg(long = "in-flight", value_name = "N")]
    in_flight: Option<usize>,
    /// Position chunks each reader decodes per read pass. Default: auto
    /// (up to 4M positions, fewer when that leaves workers short of pieces).
    #[arg(long = "decode-chunks", value_name = "N")]
    decode_chunks: Option<usize>,
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
    /// Zarr v3 codec metadata for created `values` arrays: a `codecs` JSON
    /// array, or a `{"chunk_grid": ..., "codecs": [...]}` fragment. Pass
    /// `@path` to read the JSON from a file. Replaces the default
    /// Blosc(zstd-5, byte shuffle) pipeline; express sharding as a
    /// `sharding_indexed` codec and the outer grid as `chunk_grid`.
    #[arg(
        long,
        value_name = "JSON|@FILE",
        conflicts_with_all = ["chunk_size", "column_chunk_size", "shard_size", "shard_column_size"]
    )]
    codecs: Option<String>,
    /// Comma-separated downsampling factors (e.g. 16,256), strictly ascending
    /// with no duplicates. Builds a multiscale pyramid on each imported track
    /// after the base import.
    #[arg(long, value_name = "LIST")]
    scales: Option<String>,
    /// Show import progress on stderr. This is the default; the flag is
    /// accepted so scripts can be explicit.
    #[arg(long)]
    progress: bool,
    /// Hide the import progress display.
    #[arg(long, conflicts_with = "progress")]
    no_progress: bool,
    /// Resolve the import, print the destination tracks and the layout
    /// estimate to stdout, and exit without creating or writing anything.
    #[arg(long)]
    dry_run: bool,
}

impl ImportOptions {
    /// Base pipeline config, with a progress sink labeled `label` unless
    /// `--no-progress` turned it off. The pipeline sizes the sink itself, so
    /// the label is all it needs here.
    fn config(&self, label: &str) -> Result<Config> {
        Ok(Config {
            workers: self.threads,
            in_flight_spans: self.in_flight.unwrap_or(0),
            decode_chunks: self.decode_chunks.unwrap_or(0),
            handle_budget: 0,
            chunk_size: self.chunk_size,
            column_chunk_size: self.column_chunk_size,
            shard_size: self.shard_size,
            shard_column_size: self.shard_column_size,
            column_dim: self.column_dim.clone(),
            codecs: self.parse_codecs()?,
            scales: self
                .scales
                .as_deref()
                .map(parse_scales)
                .transpose()?
                .unwrap_or_default(),
            progress: (!self.no_progress && !self.dry_run).then(|| make_sink(label)),
        })
    }

    /// Parse `--codecs`: inline JSON, or `@path` to a JSON file.
    fn parse_codecs(&self) -> Result<Option<ExplicitArraySpec>> {
        let Some(raw) = &self.codecs else {
            return Ok(None);
        };
        let text = match raw.strip_prefix('@') {
            Some(path) => std::fs::read_to_string(path)
                .with_context(|| format!("read --codecs file {path}"))?,
            None => raw.clone(),
        };
        ExplicitArraySpec::parse(&text)
            .map(Some)
            .context("parse --codecs")
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
    #[arg(long, conflicts_with_all = ["schema", "fields", "track", "dtype", "dry_run"])]
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
        Command::Scale(args) => scale_cmd(args),
    }
}

fn view(args: ViewArgs) -> Result<()> {
    init_logging(&args.global);
    view::run_view(&view::ViewSpec {
        store: args.store,
        track: args.track,
        columns: args.columns,
        output: args.output,
        no_header: args.no_header,
        threads: args.threads,
        precision: args.precision,
    })
}

fn scale_cmd(args: ScaleArgs) -> Result<()> {
    let _ = args.global.verbose;
    let factors = args.scales.as_deref().map(parse_scales).transpose()?;

    let store =
        PbzStore::open(&args.pbz).with_context(|| format!("open store {}", args.pbz.display()))?;
    let config = ScaleConfig {
        factors,
        workers: args.workers,
        ..ScaleConfig::default()
    };

    let start = Instant::now();
    let report = pbzarr::scale(&store, &args.track, &config)
        .with_context(|| format!("scale track {:?} in {}", args.track, args.pbz.display()))?;
    let elapsed = start.elapsed();

    println!(
        "scaled '{}' in {} ({} level{}):",
        args.track,
        args.pbz.display(),
        report.levels.len(),
        if report.levels.len() == 1 { "" } else { "s" }
    );
    for level in &report.levels {
        println!(
            "  factor {:>6}: {:>10} bins, {:>12} bytes",
            level.factor, level.bins, level.bytes_written
        );
    }
    println!("done in {:.2}s", elapsed.as_secs_f64());
    Ok(())
}

/// Parse a comma-separated `--scales` list into factors. Enforces strictly
/// ascending, duplicate-free order here with a clear message; the rest of
/// `ScaleConfig` validation (`>= 2`, non-empty) is left to `pbzarr::scale`.
fn parse_scales(text: &str) -> Result<Vec<u64>> {
    let mut factors = Vec::new();
    let mut prev: Option<u64> = None;
    for part in text.split(',') {
        let part = part.trim();
        let value: u64 = part.parse().with_context(|| {
            format!("invalid --scales value {part:?}: expected a non-negative integer")
        })?;
        if let Some(prev) = prev {
            if value == prev {
                bail!("--scales values must be unique (duplicate {value})");
            }
            if value < prev {
                bail!("--scales values must be strictly ascending (found {value} after {prev})");
            }
        }
        prev = Some(value);
        factors.push(value);
    }
    Ok(factors)
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
    let config = args.import.config(&label)?;
    if let Some(schema_path) = &args.bed.schema {
        info!("loading BED schema {}", schema_path.display());
        let schema = load_schema(schema_path, &sources, infer_rows)?;
        let plan = plan_bed_schema(&sources, &schema, &config).context("plan BED schema")?;
        debug!("BED schema planned against {} source(s)", sources.len());
        if args.import.dry_run {
            let tracks = plan_schema_tracks(&schema, &sources, config.column_dim.as_deref())?;
            print_dry_run(&output, &genome, &tracks, &config, sources.len());
            return Ok(());
        }
        let mut store = open_or_create(&output)?;
        execute_bed_schema_plan(&mut store, plan, genome, config).context("import BED schema")?;
    } else {
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
        if args.import.dry_run {
            let tracks = plan_bed_option_tracks(&sources, &options, config.column_dim.as_deref())?;
            print_dry_run(&output, &genome, &tracks, &config, sources.len());
            return Ok(());
        }
        let mut store = open_or_create(&output)?;
        from_bed_matrix(&mut store, &sources, genome, &options, config)
            .context("import BED columns")?;
    }
    info!("wrote {}", output.display());
    Ok(())
}

/// One destination track of a `--dry-run` preview.
struct PlannedTrack {
    name: String,
    dtype: Dtype,
    /// `Some((dimension, labels))` for a rank-2 track; `None` for scalar.
    columns: Option<(String, Vec<String>)>,
}

/// Print the destination plan and the layout estimate for a dry run.
/// Read-only: the output store is only stat'ed, never opened or created.
fn print_dry_run(
    output: &Path,
    genome: &Genome,
    tracks: &[PlannedTrack],
    config: &Config,
    readers: usize,
) {
    println!("dry run: nothing is created or written");
    println!(
        "output: {} ({})",
        output.display(),
        if output.exists() {
            "exists; a real run would add tracks to it"
        } else {
            "a real run would create it"
        }
    );
    println!(
        "genome: {} contig(s), {} positions",
        genome.contigs().len(),
        genome.contigs().iter().map(|c| c.length).sum::<u64>()
    );
    for track in tracks {
        match &track.columns {
            Some((dim, labels)) => println!(
                "track {}  {}  rank 2  columns ({dim}): {}",
                track.name,
                track.dtype.as_str(),
                labels.join(", ")
            ),
            None => println!("track {}  {}  rank 1", track.name, track.dtype.as_str()),
        }
    }
    println!();
    let shapes: Vec<TrackShape> = tracks
        .iter()
        .map(|track| TrackShape {
            name: track.name.clone(),
            dtype: track.dtype,
            columns: track
                .columns
                .as_ref()
                .map(|(_, labels)| labels.len() as u64),
        })
        .collect();
    let knobs = LayoutKnobs::resolve(
        config.chunk_size.map(|v| v as u64),
        config.column_chunk_size.map(|v| v as u64),
        config.shard_size.map(|v| v as u64),
        config.shard_column_size.map(|v| v as u64),
        config.codecs.as_ref(),
    );
    let in_flight = match config.in_flight_spans {
        0 => auto_in_flight_spans(config.workers, readers),
        n => n,
    };
    let estimate = LayoutEstimate::compute(
        GenomeGeometry {
            total_len: genome.contigs().iter().map(|c| c.length).sum(),
            contigs: genome.contigs().len() as u64,
        },
        &shapes,
        &knobs,
        readers,
        config.workers,
        in_flight,
        &config.scales,
    );
    print!("{estimate}");
}

/// Track name a schema spec resolves to: the explicit target, else the
/// header name a `Name` selector carries.
fn schema_track_name(spec: &BedColumnSpec) -> Result<&str> {
    match (&spec.track_name, &spec.selector) {
        (Some(track), _) => Ok(track),
        (None, ColumnSelector::Name(name)) => Ok(name),
        (None, ColumnSelector::Index(index)) => Err(anyhow!(
            "schema BED column {} needs a target track name",
            index + 1
        )),
    }
}

/// The tracks a schema import would create, mirroring the grouped/unique
/// layout split `plan_bed_schema` validated.
fn plan_schema_tracks(
    schema: &BedSchema,
    sources: &[pbzarr::import::Source],
    column_dim: Option<&str>,
) -> Result<Vec<PlannedTrack>> {
    let names = schema
        .0
        .iter()
        .map(schema_track_name)
        .collect::<Result<Vec<_>>>()?;
    let unique: HashSet<&str> = names.iter().copied().collect();
    if unique.len() == 1 && names.len() >= 2 {
        let layout = read_bed_layout(&sources[0].path)?;
        let labels = schema
            .0
            .iter()
            .map(|spec| match &spec.selector {
                ColumnSelector::Name(name) => name.clone(),
                ColumnSelector::Index(index) => layout
                    .header
                    .as_ref()
                    .and_then(|header| header.get(*index))
                    .cloned()
                    .unwrap_or_else(|| format!("column {}", index + 1)),
            })
            .collect();
        return Ok(vec![PlannedTrack {
            name: names[0].to_owned(),
            dtype: schema.0[0].dtype,
            columns: Some((column_dim.unwrap_or("BED column").to_owned(), labels)),
        }]);
    }
    let source_columns = (sources.len() > 1 || column_dim.is_some()).then(|| {
        (
            column_dim.unwrap_or("sample").to_owned(),
            sources
                .iter()
                .map(pbzarr::import::Source::label)
                .collect::<Vec<_>>(),
        )
    });
    Ok(schema
        .0
        .iter()
        .zip(names)
        .map(|(spec, name)| PlannedTrack {
            name: name.to_owned(),
            dtype: spec.dtype,
            columns: source_columns.clone(),
        })
        .collect())
}

/// The tracks a non-schema BED import would create, mirroring the shape
/// rules `from_bed_matrix` applies to the same options.
fn plan_bed_option_tracks(
    sources: &[pbzarr::import::Source],
    options: &BedImportOptions,
    column_dim: Option<&str>,
) -> Result<Vec<PlannedTrack>> {
    if sources.len() > 1 && column_dim.is_none() {
        bail!("2D output requires a column dimension name (--column-dim)");
    }
    let layout = read_bed_layout(&sources[0].path)?;
    let source_columns = (sources.len() > 1 || column_dim.is_some()).then(|| {
        (
            column_dim.unwrap_or("sample").to_owned(),
            sources
                .iter()
                .map(pbzarr::import::Source::label)
                .collect::<Vec<_>>(),
        )
    });
    let single = |track: &Option<String>, dtype: Dtype, need: &str| -> Result<Vec<PlannedTrack>> {
        let name = track
            .clone()
            .with_context(|| format!("a track name is required {need}"))?;
        Ok(vec![PlannedTrack {
            name,
            dtype,
            columns: source_columns.clone(),
        }])
    };
    if layout.n_cols == 3 {
        return single(&options.track, Dtype::Bool, "for BED3 input");
    }
    let Some(header) = &layout.header else {
        if layout.n_cols > 4 {
            bail!(
                "{} has {} BED value columns and no header; add a header line or use a schema",
                sources[0].path.display(),
                layout.n_cols - 3
            );
        }
        let dtype = match options.dtype {
            Some(dtype) => dtype,
            None => infer_bed_dtypes_for_sources(sources, &[3], options.infer_rows)?[0],
        };
        return single(&options.track, dtype, "for headerless input");
    };
    let selected = options
        .fields
        .clone()
        .unwrap_or_else(|| header[3..].to_vec());
    let columns = selected
        .iter()
        .map(|name| {
            header
                .iter()
                .position(|field| field == name)
                .filter(|index| *index >= 3)
                .with_context(|| {
                    format!("BED column {name:?} is not a value BED column of the header")
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let dtypes = infer_bed_dtypes_for_sources(sources, &columns, options.infer_rows)?;
    if sources.len() == 1 && selected.len() > 1 {
        if column_dim.is_none() {
            bail!("2D output requires a column dimension name (--column-dim)");
        }
        // The import itself merges raw observations across the selected
        // columns before choosing one dtype; joining the per-column dtypes
        // covers the same values but can sit one width class above it.
        let mut joined: Option<Dtype> = None;
        for dtype in dtypes {
            joined = Some(match joined {
                Some(joined) => join_dtypes(joined, dtype)?,
                None => dtype,
            });
        }
        let dtype = joined.context("no BED value columns selected")?;
        let name = options
            .track
            .clone()
            .context("a track name is required when several BED columns form one wide track")?;
        return Ok(vec![PlannedTrack {
            name,
            dtype,
            columns: Some((column_dim.unwrap_or("column").to_owned(), selected)),
        }]);
    }
    Ok(selected
        .into_iter()
        .zip(dtypes)
        .map(|(name, dtype)| PlannedTrack {
            name,
            dtype,
            columns: source_columns.clone(),
        })
        .collect())
}

/// Smallest dtype whose value range covers both inputs. Floats absorb
/// integers, matching inference (any decimal column classes as float).
fn join_dtypes(a: Dtype, b: Dtype) -> Result<Dtype> {
    if a == b {
        return Ok(a);
    }
    if a == Dtype::F64 || b == Dtype::F64 {
        return Ok(Dtype::F64);
    }
    if a == Dtype::F32 || b == Dtype::F32 {
        return Ok(Dtype::F32);
    }
    fn range(dtype: Dtype) -> (i128, i128) {
        match dtype {
            Dtype::Bool => (0, 1),
            Dtype::U8 => (0, i128::from(u8::MAX)),
            Dtype::U16 => (0, i128::from(u16::MAX)),
            Dtype::U32 => (0, i128::from(u32::MAX)),
            Dtype::I8 => (i128::from(i8::MIN), i128::from(i8::MAX)),
            Dtype::I16 => (i128::from(i16::MIN), i128::from(i16::MAX)),
            Dtype::I32 => (i128::from(i32::MIN), i128::from(i32::MAX)),
            Dtype::F32 | Dtype::F64 => unreachable!("floats are joined above"),
        }
    }
    let (lo_a, hi_a) = range(a);
    let (lo_b, hi_b) = range(b);
    let (lo, hi) = (lo_a.min(lo_b), hi_a.max(hi_b));
    if lo >= 0 {
        return Ok(if hi <= i128::from(u8::MAX) {
            Dtype::U8
        } else if hi <= i128::from(u16::MAX) {
            Dtype::U16
        } else {
            Dtype::U32
        });
    }
    if lo >= i128::from(i8::MIN) && hi <= i128::from(i8::MAX) {
        Ok(Dtype::I8)
    } else if lo >= i128::from(i16::MIN) && hi <= i128::from(i16::MAX) {
        Ok(Dtype::I16)
    } else if lo >= i128::from(i32::MIN) && hi <= i128::from(i32::MAX) {
        Ok(Dtype::I32)
    } else {
        bail!(
            "BED columns of dtypes {} and {} cannot share one wide track (range exceeds int32)",
            a.as_str(),
            b.as_str()
        )
    }
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

    let config = args.import.config(&args.bam.track)?;
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
    if args.import.dry_run {
        let source = &sources[0];
        let reader = BamReader::open(&source.path, args.bam.reference.as_deref(), mode, filter)
            .map_err(|e| anyhow!("open {}: {e}", source.path.display()))?;
        let source_columns = (n_sources > 1 || config.column_dim.is_some()).then(|| {
            (
                config
                    .column_dim
                    .clone()
                    .unwrap_or_else(|| "sample".to_owned()),
                sources
                    .iter()
                    .map(pbzarr::import::Source::label)
                    .collect::<Vec<_>>(),
            )
        });
        // Field 0 keeps the bare track name and the rest append their field
        // name, matching `from_bam`'s composition naming.
        let tracks: Vec<PlannedTrack> = reader
            .output_schema()
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| PlannedTrack {
                name: match (index, field.name.as_deref()) {
                    (0, _) | (_, None) => args.bam.track.clone(),
                    (_, Some(name)) => format!("{}_{name}", args.bam.track),
                },
                dtype: field.dtype,
                columns: source_columns.clone(),
            })
            .collect();
        print_dry_run(&output, reader.contigs(), &tracks, &config, n_sources);
        return Ok(());
    }
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

    #[test]
    fn codecs_conflicts_with_grid_flags() {
        let err = Cli::try_parse_from([
            "pbz",
            "import",
            "bed",
            "-o",
            "out.pbz",
            "--genome",
            "g.fai",
            "--track",
            "depth",
            "--codecs",
            "[]",
            "--chunk-size",
            "100",
            "in.bed.gz",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn scales_parse_ascending_list() {
        assert_eq!(parse_scales("8,64,512").unwrap(), vec![8, 64, 512]);
    }

    #[test]
    fn scales_parse_trims_whitespace() {
        assert_eq!(parse_scales(" 8 , 64 , 512 ").unwrap(), vec![8, 64, 512]);
    }

    #[test]
    fn scales_parse_single_value() {
        assert_eq!(parse_scales("8").unwrap(), vec![8]);
    }

    #[test]
    fn scales_reject_duplicates() {
        let err = parse_scales("8,8,64").unwrap_err().to_string();
        assert!(err.contains("unique"), "unexpected message: {err}");
    }

    #[test]
    fn scales_reject_non_ascending() {
        let err = parse_scales("64,8").unwrap_err().to_string();
        assert!(err.contains("ascending"), "unexpected message: {err}");
    }

    #[test]
    fn scales_reject_non_integer() {
        assert!(parse_scales("8,abc").is_err());
    }
}
