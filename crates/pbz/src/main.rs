use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, debug, info};
use pbzarr::Dataset;
use pbzarr::import::Config;
use pbzarr::import::progress::{self, make_sink};
use pbzarr::import::{Target, store_reference};
use pbzarr::io::Dtype;
use pbzarr::{ExplicitArraySpec, Reference};
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, ColumnSelector, DepthFilter, ImportMode, InferRows,
    OverlapMode, column_index_by_name, execute_bed_schema_plan, from_bam, from_bed_matrix,
    from_bigwig, from_d4, infer_bed_dtypes, infer_bed_dtypes_for_sources, plan_bed_schema,
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
    /// Compute summary statistics for one track.
    Stat(StatArgs),
    /// Import data into a PBZ store.
    Import(Box<ImportArgs>),
    /// Append columns to a track of a format-0.5 PBZ store.
    Append(Box<AppendArgs>),
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
#[command(
    after_help = "Examples:\n  pbz stat cohort.pbz depth\n  pbz stat -s median -r targets.bed cohort.pbz depth\n  pbz stat -s hist -c s1,s2 cohort.pbz af"
)]
struct StatArgs {
    #[command(flatten)]
    global: GlobalOptions,
    /// PBZ store to read.
    store: PathBuf,
    /// Track to compute over. Optional when the store holds exactly one track.
    track: Option<String>,
    /// BED file with the regions to compute over. Default: one region per
    /// chromosome.
    #[arg(short('r'), long, value_name = "BED")]
    region: Option<PathBuf>,
    /// Statistic: mean, min, max, median, or hist.
    #[arg(short('s'), long, value_name = "STAT", default_value = "mean")]
    stat: String,
    /// Column labels of a rank-2 track, comma-separated or repeated.
    /// Default: all columns in stored order.
    #[arg(short('c'), long, value_name = "LABEL", value_delimiter = ',')]
    columns: Vec<String>,
    /// Output path. A `.gz` suffix writes BGZF; default is stdout.
    #[arg(short('o'), long, value_name = "PATH")]
    output: Option<PathBuf>,
    /// Do not write the header line.
    #[arg(long)]
    no_header: bool,
    /// Worker threads for chunk decode and stat batches. Default: all cores.
    #[arg(short('t'), long, value_name = "N")]
    threads: Option<std::num::NonZeroUsize>,
    /// Significant digits for float values, `%g` style.
    #[arg(short('p'), long, value_name = "N", default_value_t = 6)]
    precision: u8,
}

#[derive(Debug, Args)]
struct ImportArgs {
    #[command(subcommand)]
    format: ImportCommand,
}

#[derive(Debug, Args)]
struct AppendArgs {
    #[command(subcommand)]
    format: AppendCommand,
}

#[derive(Debug, Subcommand)]
#[command(
    after_help = "The store fixes chunks, shards, codecs and the column dimension, so \
                  append takes no layout flags. Every root variable over the appended \
                  dimension grows; only --track receives values, and the other grown \
                  columns read as the array fill value. A variable without that \
                  dimension, such as a reduction across columns, keeps a legal shape \
                  and goes stale; pbz neither detects nor marks it."
)]
enum AppendCommand {
    /// Append columns decoded from tabix-indexed BED files.
    Bed(AppendBedArgs),
    /// Append columns decoded from BAM/CRAM files.
    Bam(AppendBamArgs),
    /// Append columns decoded from d4 files.
    D4(AppendD4Args),
    /// Append columns decoded from bigWig files.
    #[command(name = "bigwig")]
    BigWig(AppendBigWigArgs),
}

/// The store, the sources and the track every append names.
#[derive(Debug, Args)]
struct AppendTarget {
    /// Format-0.5 PBZ store to grow. It must exist.
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Track to grow. It must already have a column dimension.
    #[arg(long)]
    track: String,
}

#[derive(Debug, Args)]
struct AppendOptions {
    /// Worker threads for the append.
    #[arg(short = 't', long, default_value_t = 4)]
    threads: usize,
    /// Position chunks each reader decodes per read pass. Default: auto.
    #[arg(long = "decode-chunks", value_name = "N")]
    decode_chunks: Option<usize>,
    /// Hide the append progress display.
    #[arg(long)]
    no_progress: bool,
}

impl AppendOptions {
    /// Pipeline config for an append. `column_dim` comes from the store, so a
    /// single source still forms a column batch instead of a scalar track.
    fn config(&self, label: &str, column_dim: String) -> Config {
        Config {
            workers: self.threads,
            decode_chunks: self.decode_chunks.unwrap_or(0),
            column_dim: Some(column_dim),
            progress: (!self.no_progress).then(|| make_sink(label)),
            ..Config::default()
        }
    }
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append d4 -o cohort.pbz --track depth s3.d4 s4.d4:custom_label"
)]
struct AppendD4Args {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    append: AppendOptions,
    #[command(flatten)]
    target: AppendTarget,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append bigwig -o cohort.pbz --track signal kidney.bw:kidney"
)]
struct AppendBigWigArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    append: AppendOptions,
    #[command(flatten)]
    target: AppendTarget,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append bam -o depth.pbz --track depth s3.bam\n  pbz append bam -o comp.pbz --track depth --mode composition --fasta ref.fa s3.cram"
)]
struct AppendBamArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    append: AppendOptions,
    #[command(flatten)]
    target: AppendTarget,
    #[command(flatten)]
    decode: BamDecodeOptions,
}

/// `--schema` creates several tracks, so it has no append form.
#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append bed -o cohort.pbz --track depth s3.bed.gz s4.bed.gz"
)]
struct AppendBedArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    append: AppendOptions,
    #[command(flatten)]
    target: AppendTarget,
    /// BED header column that fills --track (requires a `#`-prefixed header).
    /// Omit when the header has one BED value column.
    #[arg(long = "field")]
    fields: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum ImportCommand {
    /// Import from tabix-indexed BED files.
    Bed(BedArgs),
    /// Import per-base depth or composition counts from BAM/CRAM files.
    Bam(BamArgs),
    /// Import per-base depth from d4 files.
    D4(D4Args),
    /// Import per-base float32 signal from bigWig files (uncovered bases are NaN).
    #[command(name = "bigwig")]
    BigWig(BigWigArgs),
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bed -o cohort.pbz --genome genome.fai --track depth --dim sample s1.bed.gz s2.bed.gz\n  pbz import bed -o mask.pbz --genome genome.fai --track callable regions.bed.gz\n  pbz import bed -o stats.pbz --genome genome.fai --dim metric --schema schema.tsv wide.bed.gz\n  pbz import bed -o cohort.pbz --genome genome.fai --track depth --dim sample --codecs '[{\"name\":\"transpose\",\"configuration\":{\"order\":[1,0]}},{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}]' s1.bed.gz s2.bed.gz"
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
    /// Worker threads for the import.
    #[arg(short = 't', long, default_value_t = 4)]
    threads: usize,
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
    /// Name of the second dimension of a 2D track. A single source with
    /// --dim still makes a 2D track. Default: `sample` for several d4,
    /// bigWig or BAM sources. BED import has no default: it needs --dim
    /// whenever the output is 2D (several sources, or several BED columns in
    /// one track).
    #[arg(short = 'c', long, value_name = "NAME")]
    dim: Option<String>,
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
    /// Hide the import progress display.
    #[arg(long)]
    no_progress: bool,
}

impl ImportOptions {
    /// Base pipeline config, with a progress sink labeled `label` unless
    /// `--no-progress` turned it off. The pipeline sizes the sink itself, so
    /// the label is all it needs here.
    fn config(&self, label: &str) -> Result<Config> {
        Ok(Config {
            workers: self.threads,
            decode_chunks: self.decode_chunks.unwrap_or(0),
            chunk_size: self.chunk_size,
            column_chunk_size: self.column_chunk_size,
            shard_size: self.shard_size,
            shard_column_size: self.shard_column_size,
            column_dim: self.dim.clone(),
            codecs: self.parse_codecs()?,
            progress: (!self.no_progress).then(|| make_sink(label)),
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
    /// PBZ output store. Import creates it when the path does not exist, and
    /// adds the track(s) to it when it exists.
    #[arg(short('o'), long, required_unless_present = "emit_schema")]
    output: Option<PathBuf>,
    #[command(flatten)]
    input: ImportInputOptions,
    /// FAI or chromosome-sizes file with the contig lengths. It is needed
    /// only when the store does not exist; an existing store gives its own.
    #[arg(long)]
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
    after_help = "Examples:\n  pbz import bam -o depth.pbz --track depth s1.bam s2.bam\n  pbz import bam -o comp.pbz --track depth --mode composition --fasta ref.fa sample.cram"
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
    /// PBZ output store. Import creates it from the source headers when the
    /// path does not exist, and adds the track(s) to it when it exists.
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Output track name. In composition mode the depth track keeps this
    /// name and the per-event tracks are named `{track}_{field}`.
    #[arg(long)]
    track: String,
    #[command(flatten)]
    decode: BamDecodeOptions,
}

/// BAM/CRAM decode flags, shared by `import bam` and `append bam`.
#[derive(Debug, Args)]
struct BamDecodeOptions {
    /// Depth counts coverage only; composition also counts per-base events,
    /// emitting `{track}_{field}` for a, c, g, t, n, ins, del, ref_skip.
    #[arg(long, value_enum, default_value_t = ImportModeArg::Depth)]
    mode: ImportModeArg,
    /// FASTA for CRAM sources (ignored for BAM).
    #[arg(long, value_name = "FASTA")]
    fasta: Option<PathBuf>,
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

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import d4 -o cohort.pbz --track depth s1.d4 s2.d4:custom_label"
)]
struct D4Args {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    d4: D4Options,
}

#[derive(Debug, Args)]
struct D4Options {
    /// PBZ output store. Import creates it from the source headers when the
    /// path does not exist, and adds the track(s) to it when it exists.
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Output track name.
    #[arg(long)]
    track: String,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bigwig -o signal.pbz --track signal sample.bw\n  pbz import bigwig -o cohort.pbz --track signal --dim sample brain.bw:brain liver.bw:liver"
)]
struct BigWigArgs {
    #[command(flatten)]
    global: GlobalOptions,
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    bigwig: BigWigOptions,
}

#[derive(Debug, Args)]
struct BigWigOptions {
    /// PBZ output store. Import creates it from the source headers when the
    /// path does not exist, and adds the track(s) to it when it exists.
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Output track name.
    #[arg(long)]
    track: String,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::View(args) => view(args),
        Command::Stat(args) => stat(args),
        Command::Import(import) => match import.format {
            ImportCommand::Bed(args) => import_bed(args),
            ImportCommand::Bam(args) => import_bam(args),
            ImportCommand::D4(args) => import_d4(args),
            ImportCommand::BigWig(args) => import_bigwig(args),
        },
        Command::Append(append) => match append.format {
            AppendCommand::Bed(args) => append_bed(args),
            AppendCommand::Bam(args) => append_bam(args),
            AppendCommand::D4(args) => append_d4(args),
            AppendCommand::BigWig(args) => append_bigwig(args),
        },
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

fn stat(args: StatArgs) -> Result<()> {
    init_logging(&args.global);
    stat::run_stat(&stat::StatSpec {
        store: args.store,
        track: args.track,
        region: args.region,
        stat: args.stat,
        columns: args.columns,
        output: args.output,
        no_header: args.no_header,
        threads: args.threads,
        precision: args.precision,
    })
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

    let output = args.bed.output.context("--output is required to import")?;
    let genome = args
        .bed
        .genome
        .as_deref()
        .map(|path| {
            Reference::from_fai(path).with_context(|| format!("read genome {}", path.display()))
        })
        .transpose()?;
    let genome = store_reference(&output, genome)?;
    info!("genome: {} contig(s)", genome.contigs().len());
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
        execute_bed_schema_plan(output.as_path(), plan, genome, config)
            .context("import BED schema")?;
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
        from_bed_matrix(output.as_path(), &sources, genome, &options, config)
            .context("import BED columns")?;
    }
    info!("wrote {}", output.display());
    Ok(())
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
    let mode = match args.bam.decode.mode {
        ImportModeArg::Depth => ImportMode::Depth,
        ImportModeArg::Composition => ImportMode::Composition,
    };
    let overlap = match args.bam.decode.overlap {
        OverlapModeArg::Proper => OverlapMode::ProperOnly,
        OverlapModeArg::All => OverlapMode::All,
        OverlapModeArg::None => OverlapMode::None,
    };
    let filter = DepthFilter {
        min_mapq: args.bam.decode.min_mapq,
        exclude_flags: args.bam.decode.exclude_flags,
        min_bq: args.bam.decode.min_bq,
        overlap,
        count_deletions: args.bam.decode.count_deletions,
    };

    let output = args.bam.output;
    let report = from_bam(
        output.as_path(),
        &args.bam.track,
        &sources,
        mode,
        filter,
        args.bam.decode.fasta,
        config,
    )
    .context("import BAM/CRAM")?;

    print_report("imported", n_sources, &output, &report);
    Ok(())
}

fn import_d4(args: D4Args) -> Result<()> {
    init_logging(&args.global);
    let sources: Vec<pbzarr::import::Source> = args
        .d4
        .input
        .resolve_input()?
        .into_iter()
        .map(|source| pbzarr::import::Source {
            path: source.path,
            column_label: source.label,
        })
        .collect();
    let n_sources = sources.len();
    debug!("resolved {n_sources} d4 source(s)");

    let config = args.import.config(&args.d4.track)?;
    let output = args.d4.output;
    let report =
        from_d4(output.as_path(), &args.d4.track, &sources, config).context("import d4")?;

    print_report("imported", n_sources, &output, &report);
    Ok(())
}

fn import_bigwig(args: BigWigArgs) -> Result<()> {
    init_logging(&args.global);
    let sources: Vec<pbzarr::import::Source> = args
        .bigwig
        .input
        .resolve_input()?
        .into_iter()
        .map(|source| pbzarr::import::Source {
            path: source.path,
            column_label: source.label,
        })
        .collect();
    let n_sources = sources.len();
    debug!("resolved {n_sources} bigWig source(s)");

    let config = args.import.config(&args.bigwig.track)?;
    let output = args.bigwig.output;
    let report = from_bigwig(output.as_path(), &args.bigwig.track, &sources, config)
        .context("import bigWig")?;

    print_report("imported", n_sources, &output, &report);
    Ok(())
}

/// Resolve the sources an append decodes, in column order.
fn append_sources(input: &ImportInputOptions) -> Result<Vec<pbzarr::import::Source>> {
    Ok(input
        .resolve_input()?
        .into_iter()
        .map(|source| pbzarr::import::Source {
            path: source.path,
            column_label: source.label,
        })
        .collect())
}

/// Open the store an append grows and read the column dimension of `track`.
///
/// The dimension comes from the store so that one source still forms a column
/// batch; an append never names a new dimension.
fn open_append(path: &Path, track: &str) -> Result<(Dataset, String)> {
    if !path.exists() {
        bail!(
            "append needs an existing store; {} does not exist",
            path.display()
        );
    }
    let dataset = Dataset::open(path).with_context(|| format!("open store {}", path.display()))?;
    let info = dataset
        .variables()
        .get(track)
        .ok_or_else(|| anyhow!("track {track:?} is not in {}", path.display()))?;
    let dim = info.dims.get(1).cloned().ok_or_else(|| {
        anyhow!(
            "track {track:?} has dimension(s) {:?}; append grows a column dimension",
            info.dims
        )
    })?;
    Ok((dataset, dim))
}

fn report_append(target: &AppendTarget, sources: usize, report: &pbzarr::import::Report) {
    print_report("appended", sources, &target.output, report);
}

fn print_report(verb: &str, sources: usize, path: &Path, report: &pbzarr::import::Report) {
    debug!(
        "wrote {} bytes in {:.1}s",
        report.bytes_written, report.wall_seconds
    );
    println!(
        "{verb} {sources} sources -> {} ({} positions, {} chunks written, {} skipped)",
        path.display(),
        report.positions,
        report.chunks_written,
        report.chunks_skipped
    );
}

fn append_d4(args: AppendD4Args) -> Result<()> {
    init_logging(&args.global);
    let sources = append_sources(&args.target.input)?;
    let (mut dataset, dim) = open_append(&args.target.output, &args.target.track)?;
    let config = args.append.config(&args.target.track, dim);
    let report = from_d4(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        config,
    )
    .context("append d4")?;
    report_append(&args.target, sources.len(), &report);
    Ok(())
}

fn append_bigwig(args: AppendBigWigArgs) -> Result<()> {
    init_logging(&args.global);
    let sources = append_sources(&args.target.input)?;
    let (mut dataset, dim) = open_append(&args.target.output, &args.target.track)?;
    let config = args.append.config(&args.target.track, dim);
    let report = from_bigwig(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        config,
    )
    .context("append bigWig")?;
    report_append(&args.target, sources.len(), &report);
    Ok(())
}

fn append_bam(args: AppendBamArgs) -> Result<()> {
    init_logging(&args.global);
    let sources = append_sources(&args.target.input)?;
    let (mut dataset, dim) = open_append(&args.target.output, &args.target.track)?;
    let config = args.append.config(&args.target.track, dim);
    let mode = match args.decode.mode {
        ImportModeArg::Depth => ImportMode::Depth,
        ImportModeArg::Composition => ImportMode::Composition,
    };
    let filter = DepthFilter {
        min_mapq: args.decode.min_mapq,
        exclude_flags: args.decode.exclude_flags,
        min_bq: args.decode.min_bq,
        overlap: match args.decode.overlap {
            OverlapModeArg::Proper => OverlapMode::ProperOnly,
            OverlapModeArg::All => OverlapMode::All,
            OverlapModeArg::None => OverlapMode::None,
        },
        count_deletions: args.decode.count_deletions,
    };
    let report = from_bam(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        mode,
        filter,
        args.decode.fasta,
        config,
    )
    .context("append BAM/CRAM")?;
    report_append(&args.target, sources.len(), &report);
    Ok(())
}

fn append_bed(args: AppendBedArgs) -> Result<()> {
    init_logging(&args.global);
    let sources = append_sources(&args.target.input)?;
    let (mut dataset, dim) = open_append(&args.target.output, &args.target.track)?;
    let genome = dataset.reference()?.clone();
    let dtype = Dtype::from_str(
        dataset.variables()[&args.target.track].metadata["data_type"]
            .as_str()
            .unwrap_or(""),
    )
    .with_context(|| format!("read the dtype of track {:?}", args.target.track))?;
    let config = args.append.config(&args.target.track, dim);
    let options = BedImportOptions {
        fields: (!args.fields.is_empty()).then_some(args.fields),
        track: Some(args.target.track.clone()),
        dtype: Some(dtype),
        append: true,
        ..BedImportOptions::default()
    };
    let report = from_bed_matrix(
        Target::Append(&mut dataset),
        &sources,
        genome,
        &options,
        config,
    )
    .context("append BED columns")?;
    report_append(&args.target, sources.len(), &report);
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
    fn removed_knobs_are_rejected() {
        let import = |flag: &[&str]| {
            let mut args = vec!["pbz", "import", "d4", "-o", "out.pbz", "--track", "depth"];
            args.extend_from_slice(flag);
            args.push("in.d4");
            Cli::try_parse_from(args)
        };
        assert!(import(&[]).is_ok());
        assert!(import(&["--no-progress", "--decode-chunks", "4"]).is_ok());
        for flag in [
            &["--in-flight", "8"][..],
            &["--scales", "16"],
            &["--progress"],
        ] {
            let err = import(flag).unwrap_err();
            assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
        }
        let err = Cli::try_parse_from(["pbz", "scale", "-z", "s.pbz", "-t", "depth"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
    }

    #[test]
    fn dim_and_fasta_replace_the_old_flags() {
        let bam = |flags: &[&str]| {
            let mut args = vec!["pbz", "import", "bam", "-o", "out.pbz", "--track", "depth"];
            args.extend_from_slice(flags);
            args.push("in.cram");
            Cli::try_parse_from(args)
        };
        assert!(bam(&["--dim", "sample", "--fasta", "ref.fa", "--threads", "2"]).is_ok());
        for flag in [&["--column-dim", "sample"][..], &["--reference", "ref.fa"]] {
            let err = bam(flag).unwrap_err();
            assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
        }
    }
}
