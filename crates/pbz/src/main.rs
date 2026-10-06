use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, debug, info, warn};
use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::Config;
use pbzarr::import::Target;
use pbzarr::import::progress::{self, make_sink};
use pbzarr::io::{Dtype, ValueReader as _};
use pbzarr::{Contig, ExplicitArraySpec, Genome, PbzError, PbzStore, ScaleConfig};
use pbzarr_readers::{
    BamReader, BedColumnSpec, BedImportOptions, BedSchema, BigWigReader, ColumnSelector, D4Reader,
    DepthFilter, ImportMode, InferRows, OverlapMode, column_index_by_name, execute_bed_schema_plan,
    from_bam, from_bed_matrix, from_bigwig, from_d4, infer_bed_dtypes,
    infer_bed_dtypes_for_sources, plan_bed_schema, read_bed_layout,
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
    /// Number of append workers.
    #[arg(short = 't', long, default_value_t = 4)]
    threads: usize,
    /// Position spans open at once in the import engine. Default: auto.
    #[arg(long = "in-flight", value_name = "N")]
    in_flight: Option<usize>,
    /// Position chunks each reader decodes per read pass. Default: auto.
    #[arg(long = "decode-chunks", value_name = "N")]
    decode_chunks: Option<usize>,
    /// Show append progress on stderr. This is the default; the flag is
    /// accepted so scripts can be explicit.
    #[arg(long)]
    progress: bool,
    /// Hide the append progress display.
    #[arg(long, conflicts_with = "progress")]
    no_progress: bool,
}

impl AppendOptions {
    /// Pipeline config for an append. `column_dim` comes from the store, so a
    /// single source still forms a column batch instead of a scalar track.
    fn config(&self, label: &str, column_dim: String) -> Config {
        Config {
            workers: self.threads,
            in_flight_spans: self.in_flight.unwrap_or(0),
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
    after_help = "Examples:\n  pbz append bam -o depth.pbz --track depth s3.bam\n  pbz append bam -o comp.pbz --track depth --mode composition --reference ref.fa s3.cram"
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
    /// BED header column to decode (requires a `#`-prefixed header). Omit to
    /// decode all BED value columns.
    #[arg(long = "field")]
    fields: Vec<String>,
    /// Records sampled for dtype inference, or `all`.
    #[arg(long, default_value = "1000", value_name = "N|all")]
    infer_rows: String,
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
    /// Show scale progress on stderr. This is the default; the flag is
    /// accepted for symmetry with --no-progress.
    #[arg(long)]
    progress: bool,
    /// Hide the scale progress display.
    #[arg(long, conflicts_with = "progress")]
    no_progress: bool,
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
    /// PBZ output store. It is created when absent.
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
    after_help = "Examples:\n  pbz import bigwig -o signal.pbz --track signal sample.bw\n  pbz import bigwig -o cohort.pbz --track signal -c sample brain.bw:brain liver.bw:liver"
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
    /// PBZ output store. It is created when absent.
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

fn scale_cmd(args: ScaleArgs) -> Result<()> {
    init_logging(&args.global);
    let factors = args.scales.as_deref().map(parse_scales).transpose()?;

    let store =
        PbzStore::open(&args.pbz).with_context(|| format!("open store {}", args.pbz.display()))?;
    let config = ScaleConfig {
        factors,
        workers: args.workers,
        progress: (!args.no_progress).then(|| make_sink(&format!("scale {}", args.track))),
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
        let mut destination = open_or_create(&output, args.import.scales.is_some(), || {
            Ok(genome.contigs().to_vec())
        })?;
        execute_bed_schema_plan(destination.target(), plan, genome, config)
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
        let mut destination = open_or_create(&output, args.import.scales.is_some(), || {
            Ok(genome.contigs().to_vec())
        })?;
        from_bed_matrix(destination.target(), &sources, genome, &options, config)
            .context("import BED columns")?;
    }
    info!("wrote {}", output.display());
    Ok(())
}

/// An opened import destination, holding whichever store the chosen format
/// writes. `target` hands it to the format entry points.
enum Output {
    Store(PbzStore),
    Dataset(Dataset),
}

impl Output {
    fn target(&mut self) -> Target<'_> {
        match self {
            Self::Store(store) => Target::Store(store),
            Self::Dataset(dataset) => Target::Dataset(dataset),
        }
    }
}

/// Open `path` in `format`, creating it when it does not exist yet. Importing
/// into an existing store adds tracks to it, which is easy to do by accident
/// with a stale path, so say which of the two happened. `reference` supplies
/// the ordered contigs a new format-0.5 Dataset fixes at creation; it is only
/// called when such a Dataset is created.
fn open_or_create(
    path: &Path,
    scales_requested: bool,
    reference: impl FnOnce() -> Result<Vec<Contig>>,
) -> Result<Output> {
    if path.exists() {
        warn!(
            "{} exists; adding track(s) to the existing store",
            path.display()
        );
        return match Dataset::open(path) {
            Ok(dataset) => {
                if scales_requested {
                    bail!(
                        "--scales is not supported by format 0.5: a multiscale pyramid under one root Dataset is not defined"
                    );
                }
                Ok(Output::Dataset(dataset))
            }
            Err(PbzError::UnsupportedFormat(_)) => PbzStore::open(path)
                .map(Output::Store)
                .with_context(|| format!("open output store {}", path.display())),
            Err(error) => {
                Err(error).with_context(|| format!("open output dataset {}", path.display()))
            }
        };
    }
    if scales_requested {
        bail!(
            "--scales is not supported by format 0.5: a multiscale pyramid under one root Dataset is not defined"
        );
    }
    debug!("creating store {}", path.display());
    let reference = Reference::new(reference()?)?;
    Dataset::create(path, reference)
        .map(Output::Dataset)
        .with_context(|| format!("create output dataset {}", path.display()))
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
    let mut destination = open_or_create(&output, args.import.scales.is_some(), || {
        let source = &sources[0];
        let reader = BamReader::open(
            &source.path,
            args.bam.decode.reference.as_deref(),
            mode,
            filter,
        )
        .map_err(|e| anyhow!("open {}: {e}", source.path.display()))?;
        Ok(reader.contigs().contigs().to_vec())
    })?;

    let report = from_bam(
        destination.target(),
        &args.bam.track,
        &sources,
        mode,
        filter,
        args.bam.decode.reference,
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

    let mut destination = open_or_create(&output, args.import.scales.is_some(), || {
        let source = &sources[0];
        let reader = D4Reader::open(&source.path)
            .map_err(|e| anyhow!("open {}: {e}", source.path.display()))?;
        Ok(reader.contigs().contigs().to_vec())
    })?;

    let report =
        from_d4(destination.target(), &args.d4.track, &sources, config).context("import d4")?;

    debug!(
        "d4 import wrote {} bytes across {} contig(s)",
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

    let mut destination = open_or_create(&output, args.import.scales.is_some(), || {
        let source = &sources[0];
        let reader = BigWigReader::open(&source.path)
            .map_err(|e| anyhow!("open {}: {e}", source.path.display()))?;
        Ok(reader.contigs().contigs().to_vec())
    })?;

    let report = from_bigwig(destination.target(), &args.bigwig.track, &sources, config)
        .context("import bigWig")?;

    debug!(
        "bigWig import wrote {} bytes across {} contig(s)",
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
    let dataset = match Dataset::open(path) {
        Ok(dataset) => dataset,
        Err(PbzError::UnsupportedFormat(found)) => bail!(
            "{} is not a format 0.5 store (found {found}); pbz append only grows format 0.5",
            path.display()
        ),
        Err(error) => {
            return Err(error).with_context(|| format!("open store {}", path.display()));
        }
    };
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
    debug!(
        "append wrote {} bytes across {} contig(s)",
        report.bytes_written, report.contigs_written
    );
    println!(
        "appended {sources} sources -> {} ({} tasks, {} skipped)",
        target.output.display(),
        report.tasks_completed,
        report.tasks_skipped
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
        args.decode.reference,
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
    let genome = Genome::new(dataset.reference()?.contigs().to_vec())?;
    let config = args.append.config(&args.target.track, dim);
    let options = BedImportOptions {
        fields: (!args.fields.is_empty()).then_some(args.fields),
        track: Some(args.target.track.clone()),
        infer_rows: parse_infer_rows(&args.infer_rows)?,
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
