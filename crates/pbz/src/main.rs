use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use indicatif_log_bridge::LogWrapper;
use log::{LevelFilter, debug, info};
use pbzarr::Dataset;
use pbzarr::import::Source;
use pbzarr::import::progress;
use pbzarr::import::{Config, ConfigBuilder};
use pbzarr::import::{Target, store_reference};
use pbzarr::io::Dtype;
use pbzarr::stat::StatKind;
use pbzarr::{ExplicitArraySpec, Reference};
use pbzarr_readers::{
    BedColumnRequest, BedImportOptions, BedSchema, DepthFilter, ImportMode, InferRows, OverlapMode,
    build_bed_schema, execute_bed_schema_plan, from_bam, from_bed_matrix, from_bigwig, from_d4,
    infer_bed_dtypes, plan_bed_schema, read_bed_layout,
};

mod fmt;
mod info;
mod output;
mod regions;
mod stat;
mod validate;
mod view;

#[derive(Debug, Parser)]
#[command(name = "pbz", version, about = "Per-base Zarr tools")]
struct Cli {
    #[command(flatten)]
    global: GlobalOptions,
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
    /// Print the reference, dimensions and per-variable layout of a store.
    Info(InfoArgs),
    /// Check a store against the PBZ format rules. Exit code 1 when the
    /// store is not valid.
    Validate(ValidateArgs),
}

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  pbz info cohort.pbz\n  pbz info cohort.pbz.zip --json\n\nThe chunk column gives the inner chunk shape. The shard column gives the shard shape, or - when the array is not sharded. stored and objects count the chunk or shard objects, without the zarr.json metadata."
)]
struct InfoArgs {
    /// PBZ store to describe (directory or zip).
    store: PathBuf,
    /// Print one JSON object instead of text.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  pbz validate cohort.pbz\n  pbz validate cohort.pbz.zip --json\n\nThe check reads metadata, the reference arrays and the dimension labels. It does not read track values."
)]
struct ValidateArgs {
    /// PBZ store to check (directory or zip).
    store: PathBuf,
    /// Print one JSON object with the status and the problems. Each problem
    /// also gives the code of the format rule it breaks.
    #[arg(long)]
    json: bool,
}

/// The track shapes that `view` and `stat` read.
macro_rules! track_limits {
    () => {
        "The track must have position as its first dimension and at most one other \
         dimension. Tracks on a bin_N dimension are not supported. A track that declares \
         missing values with _FillValue must be a float track with a NaN fill."
    };
}

#[derive(Debug, Args)]
#[command(
    after_help = concat!(
        "Examples:\n  pbz view cohort.pbz --track depth\n  pbz view cohort.pbz --track depth -r chr1:0-1000000 -o depth.bedgraph.gz\n  pbz view cohort.pbz --track af -c s1,s2 -R targets.bed\n\nEach region is written in order. Rows do not extend across region edges.\n\n",
        track_limits!()
    )
)]
struct ViewArgs {
    /// PBZ store to export.
    store: PathBuf,
    /// Track to export. Optional when the store holds exactly one track.
    #[arg(long)]
    track: Option<String>,
    #[command(flatten)]
    regions: RegionArgs,
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
    after_help = concat!(
        "Examples:\n  pbz stat cohort.pbz --track depth\n  pbz stat cohort.pbz --track depth -s median -R targets.bed\n  pbz stat cohort.pbz --track depth -r chr1:0-1000 -r chr2:500-900\n  pbz stat cohort.pbz --track af -s hist -c s1,s2\n\nWithout a region, stat writes one row per contig.\n\n",
        track_limits!()
    )
)]
struct StatArgs {
    /// PBZ store to read.
    store: PathBuf,
    /// Track to compute over. Optional when the store holds exactly one track.
    #[arg(long)]
    track: Option<String>,
    #[command(flatten)]
    regions: RegionArgs,
    /// Statistic to compute.
    #[arg(short('s'), long, value_enum, default_value_t = StatArg::Mean)]
    stat: StatArg,
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
struct RegionArgs {
    /// Region CHROM:START-END, 0-based and half-open as in BED
    /// (chr1:0-100 is the first 100 bases). CHROM alone selects the whole
    /// contig, also when the name contains ':'. An end past the contig is an
    /// error. Repeat for more regions.
    /// Without -r or -R, the command reads the whole genome.
    #[arg(short('r'), long = "region", value_name = "REGION")]
    region: Vec<String>,
    /// BED file (plain, gzip, or bgzip) with regions to read after the -r
    /// regions. The first three tab-separated columns give CHROM, START,
    /// END, 0-based and half-open. Lines that start with #, track, or
    /// browser are skipped.
    #[arg(short('R'), long = "regions", value_name = "BED")]
    regions: Option<PathBuf>,
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
                  dimension grows. The tracks that the append writes get values in the \
                  new columns: --track, and with bam --mode composition also the \
                  {track}_{field} tracks. In the other variables, the new columns read \
                  as the fill value. A variable without that dimension, such as a \
                  reduction across columns, keeps its shape and becomes stale. pbz does \
                  not detect or mark it."
)]
enum AppendCommand {
    /// Append columns decoded from tabix-indexed BED files.
    Bed(AppendBedArgs),
    /// Append columns decoded from BAM/CRAM files.
    Bam(AppendBamArgs),
    /// Append columns decoded from d4 files.
    #[command(
        arg_required_else_help = true,
        after_help = "Examples:\n  pbz append d4 cohort.pbz s3.d4 s4.d4:custom_label --track depth"
    )]
    D4(AppendSignalArgs),
    /// Append columns decoded from bigWig files.
    #[command(
        name = "bigwig",
        arg_required_else_help = true,
        after_help = "Examples:\n  pbz append bigwig cohort.pbz kidney.bw:kidney --track signal"
    )]
    BigWig(AppendSignalArgs),
}

/// The store, the sources and the track every append names.
#[derive(Debug, Args)]
struct AppendTarget {
    /// PBZ store to grow. It must exist.
    store: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Track to grow. It must already have a column dimension.
    #[arg(long)]
    track: String,
}

/// Flags that import and append share.
#[derive(Debug, Args)]
struct RunOptions {
    /// Worker threads for decode and write.
    #[arg(short = 't', long, default_value_t = 4)]
    threads: usize,
    /// Position chunks each reader decodes per read pass. Default: auto
    /// (up to 4M positions, fewer when that leaves workers short of pieces).
    #[arg(long = "decode-chunks", value_name = "N")]
    decode_chunks: Option<usize>,
    /// Hide the progress display.
    #[arg(long)]
    no_progress: bool,
}

impl RunOptions {
    /// A progress sink labeled `label` unless `--no-progress` or `quiet` turned
    /// it off. The pipeline sizes the sink itself.
    fn builder(&self, label: &str, quiet: bool) -> ConfigBuilder {
        ConfigBuilder {
            workers: Some(self.threads),
            decode_chunks: self.decode_chunks,
            progress: (!self.no_progress && !quiet).then(|| label.to_owned()),
            ..ConfigBuilder::default()
        }
    }

    /// Pipeline config for an append. `column_dim` comes from the store, so a
    /// single source still forms a column batch instead of a scalar track.
    fn append_config(&self, label: &str, column_dim: String, quiet: bool) -> Config {
        ConfigBuilder {
            column_dim: Some(column_dim),
            ..self.builder(label, quiet)
        }
        .build()
    }
}

/// The d4 and bigWig appends take no format flags.
#[derive(Debug, Args)]
struct AppendSignalArgs {
    #[command(flatten)]
    run: RunOptions,
    #[command(flatten)]
    target: AppendTarget,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append bam depth.pbz s3.bam --track depth\n  pbz append bam comp.pbz s3.cram --track depth --mode composition --fasta ref.fa"
)]
struct AppendBamArgs {
    #[command(flatten)]
    run: RunOptions,
    #[command(flatten)]
    target: AppendTarget,
    #[command(flatten)]
    decode: BamDecodeOptions,
}

/// `--schema` creates several tracks, so it has no append form.
#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz append bed cohort.pbz s3.bed.gz s4.bed.gz --track depth"
)]
struct AppendBedArgs {
    #[command(flatten)]
    run: RunOptions,
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
    #[command(
        arg_required_else_help = true,
        after_help = "Examples:\n  pbz import d4 cohort.pbz s1.d4 s2.d4:custom_label --track depth"
    )]
    D4(SignalArgs),
    /// Import per-base float32 signal from bigWig files (uncovered bases are NaN).
    #[command(
        name = "bigwig",
        arg_required_else_help = true,
        after_help = "Examples:\n  pbz import bigwig signal.pbz sample.bw --track signal\n  pbz import bigwig cohort.pbz brain.bw:brain liver.bw:liver --track signal --dim sample"
    )]
    BigWig(SignalArgs),
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    mut_arg("input_files", |arg| arg.required_unless_present_any(["emit_schema"])),
    after_help = "Examples:\n  pbz import bed cohort.pbz s1.bed.gz s2.bed.gz --genome genome.fai --track depth --dim sample\n  pbz import bed mask.pbz regions.bed.gz --genome genome.fai --track callable\n  pbz import bed --emit-schema wide.bed.gz > schema.tsv\n  pbz import bed stats.pbz wide.bed.gz --genome genome.fai --dim metric --schema schema.tsv\n\nTracks without --schema (S sources, V selected BED value columns):\n  BED3            one bool track, true where an interval covers the base. Needs --track,\n                  and --dim for several sources.\n  S = 1, V = 1    one track (position). Needs --track when the file has no header.\n  S = 1, V > 1    one track (position, V). The header names are the column labels.\n                  Needs --track and --dim.\n  S > 1, V = 1    one track (position, S). The source labels are the column labels.\n                  Needs --dim, and --track when the files have no header.\n  S > 1, V > 1    one track (position, S) for each value column, with the name of\n                  that column. Needs --dim.\nA file without a `#chrom start end ...` header can have one value column only.\n--dim with one source and one value column also makes a (position, S) track.\nBases that no interval covers read as 0 (false for bool)."
)]
struct BedArgs {
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    bed: BedOptions,
}

#[derive(Debug, Args)]
struct GlobalOptions {
    /// Increase logging verbosity: -v info, -vv debug, -vvv trace. Warnings
    /// and errors always print. `RUST_LOG` overrides this when set.
    #[arg(short, long, action = ArgAction::Count, global = true)]
    verbose: u8,
    /// Hide the summary line and the progress display of import and append.
    #[arg(short, long, global = true)]
    quiet: bool,
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
    #[command(flatten)]
    run: RunOptions,
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
    #[arg(long, value_name = "NAME")]
    dim: Option<String>,
    /// Zarr v3 codec metadata for the created track arrays: a `codecs` JSON
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
    /// Write a new single-file zip store. A STORE path that ends in
    /// `.zip` also does this. The path must not exist. Without a
    /// shard flag or --codecs, each shard holds 64 position chunks and all
    /// columns.
    #[arg(long)]
    zip: bool,
}

fn has_zip_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
}

impl ImportOptions {
    /// The import destination for `output`: a new zip store when --zip is
    /// given or the path ends in `.zip`, otherwise a directory store.
    fn target<'a>(&self, output: &'a Path) -> Result<Target<'a>> {
        let zip = self.zip || has_zip_extension(output);
        if !zip {
            return Ok(Target::Path(output));
        }
        if output.exists() {
            bail!(
                "{} exists. A zip store is written once, so give a new path",
                output.display()
            );
        }
        Ok(Target::Zip(output))
    }

    fn config(&self, label: &str, quiet: bool) -> Result<Config> {
        Ok(ConfigBuilder {
            chunk_size: self.chunk_size,
            column_chunk_size: self.column_chunk_size,
            shard_size: self.shard_size,
            shard_column_size: self.shard_column_size,
            column_dim: self.dim.clone(),
            codecs: self.parse_codecs()?,
            ..self.run.builder(label, quiet)
        }
        .build())
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
#[group(multiple = false)]
struct ImportInputOptions {
    /// Input files, optionally suffixed with `:LABEL`. The default label is the
    /// file name without its format extension (`s1.bed.gz` gives `s1`).
    #[arg(
        value_name = "PATH[:LABEL]",
        value_parser = parse_source,
        required_unless_present = "file_list"
    )]
    input_files: Vec<Source>,
    /// Tab-delimited source list with PATH or PATH<TAB>LABEL records. Relative
    /// paths start from the directory of the list file.
    #[arg(short, long, value_name = "PATH")]
    file_list: Option<PathBuf>,
}

/// Parse `PATH[:LABEL]`. The text after the last `:` is a label only when it
/// has no path separator, so URLs and Windows paths stay whole.
fn parse_source(value: &str) -> std::result::Result<Source, String> {
    let (path, label) = match value.rsplit_once(':') {
        Some((path, label)) if !label.contains(['/', '\\']) => (path, Some(label)),
        _ => (value, None),
    };
    if path.is_empty() {
        return Err("source path must not be empty".into());
    }
    if label == Some("") {
        return Err("source label must not be empty".into());
    }
    Ok(Source {
        path: PathBuf::from(path),
        column_label: label.map(str::to_owned),
    })
}

fn read_source_list(path: &Path, format: &str) -> Result<Vec<Source>> {
    let file = File::open(path).with_context(|| format!("open source list {}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new(""));
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
        let column_label = fields.get(1).map(|label| label.to_string());
        if column_label.as_deref() == Some("") {
            bail!(
                "source list {} line {} has an empty label",
                path.display(),
                index + 1
            );
        }
        let source_path = if fields[0].contains("://") {
            PathBuf::from(fields[0])
        } else {
            base.join(fields[0])
        };
        sources.push(Source {
            path: source_path,
            column_label,
        });
    }

    if sources.is_empty() {
        bail!(
            "source list {} contains no {format} sources",
            path.display()
        );
    }
    Ok(sources)
}

impl ImportInputOptions {
    /// The sources in column order, from the command line or the source list.
    fn resolve_input(&self, format: &str) -> Result<Vec<Source>> {
        let sources = match &self.file_list {
            Some(path) => read_source_list(path, format)?,
            None => self.input_files.clone(),
        };
        if sources.is_empty() {
            bail!("at least one input source is required");
        }
        Source::check_labels(&sources)?;
        Ok(sources)
    }
}

#[derive(Debug, Args)]
struct BedOptions {
    /// PBZ store to write. Import creates it when the path does not exist,
    /// and adds the track(s) to it when it exists.
    #[arg(required_unless_present = "emit_schema")]
    store: Option<PathBuf>,
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
    #[arg(long, value_enum, conflicts_with = "schema")]
    dtype: Option<DtypeArg>,
    /// TSV schema mapping BED columns to target tracks. Unique target names
    /// create one track per BED column; repeated one-target rows create a
    /// single BED-column-axis track. A BED column is a header name or 1-based
    /// column index.
    #[arg(long, value_name = "PATH")]
    schema: Option<PathBuf>,
    /// Print a schema TSV for this BED file (BED column, target track,
    /// inferred dtype) to stdout and exit without importing. Edit it and feed
    /// it back via --schema.
    #[arg(
        long,
        value_name = "BED",
        conflicts_with_all = [
            "store", "input_files", "file_list", "genome", "schema", "fields", "track", "dtype"
        ]
    )]
    emit_schema: Option<PathBuf>,
    /// Records sampled for dtype inference (default 1000), or `all`. A sample
    /// infers conservative classes (bool/int32/float32); `all` (or a file
    /// smaller than the sample) scans exhaustively and min-widths (uint8,
    /// int16, ...).
    #[arg(long, value_name = "N|all")]
    infer_rows: Option<String>,
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bam depth.pbz s1.bam s2.bam --track depth\n  pbz import bam comp.pbz sample.cram --track depth --mode composition --fasta ref.fa",
    mut_arg("track", |arg| arg.help(
        "Output track name. In composition mode the depth track keeps this name and the \
         per-event tracks are named `{track}_{field}`"
    ))
)]
struct BamArgs {
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    target: ImportTarget,
    #[command(flatten)]
    decode: BamDecodeOptions,
}

/// The store, the sources and the track of a d4, bigWig or BAM import.
#[derive(Debug, Args)]
struct ImportTarget {
    /// PBZ store to write. Import creates it from the source headers when
    /// the path does not exist, and adds the track(s) to it when it exists.
    store: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// Output track name.
    #[arg(long)]
    track: String,
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
    /// SAM flag bits that exclude a read: a decimal or 0x hex number, or
    /// comma-separated samtools flag names (PAIRED, PROPER_PAIR, UNMAP,
    /// MUNMAP, REVERSE, MREVERSE, READ1, READ2, SECONDARY, QCFAIL, DUP,
    /// SUPPLEMENTARY).
    #[arg(
        long,
        value_name = "FLAGS",
        value_parser = parse_sam_flags,
        default_value = "UNMAP,SECONDARY,QCFAIL,DUP"
    )]
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

const SAM_FLAGS: [(&str, u16); 12] = [
    ("PAIRED", 0x1),
    ("PROPER_PAIR", 0x2),
    ("UNMAP", 0x4),
    ("MUNMAP", 0x8),
    ("REVERSE", 0x10),
    ("MREVERSE", 0x20),
    ("READ1", 0x40),
    ("READ2", 0x80),
    ("SECONDARY", 0x100),
    ("QCFAIL", 0x200),
    ("DUP", 0x400),
    ("SUPPLEMENTARY", 0x800),
];

/// Parse SAM flags as samtools does: `1796`, `0x704`, or
/// `UNMAP,SECONDARY,QCFAIL,DUP`.
fn parse_sam_flags(value: &str) -> std::result::Result<u16, String> {
    let mut flags = 0;
    for item in value.split(',').map(str::trim) {
        let bits = if let Some(hex) = item.strip_prefix("0x").or(item.strip_prefix("0X")) {
            u16::from_str_radix(hex, 16).ok()
        } else if item.bytes().all(|b| b.is_ascii_digit()) {
            item.parse().ok()
        } else {
            SAM_FLAGS
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(item))
                .map(|&(_, bit)| bit)
        };
        flags |= bits
            .ok_or_else(|| format!("{item:?} is not a SAM flag number or a samtools flag name"))?;
    }
    Ok(flags)
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum StatArg {
    Mean,
    Min,
    Max,
    Median,
    Hist,
}

impl From<StatArg> for StatKind {
    fn from(stat: StatArg) -> Self {
        match stat {
            StatArg::Mean => StatKind::Mean,
            StatArg::Min => StatKind::Min,
            StatArg::Max => StatKind::Max,
            StatArg::Median => StatKind::Median,
            StatArg::Hist => StatKind::Hist,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DtypeArg {
    Bool,
    #[value(name = "uint8")]
    U8,
    #[value(name = "uint16")]
    U16,
    #[value(name = "uint32")]
    U32,
    #[value(name = "uint64")]
    U64,
    #[value(name = "int8")]
    I8,
    #[value(name = "int16")]
    I16,
    #[value(name = "int32")]
    I32,
    #[value(name = "int64")]
    I64,
    #[value(name = "float32")]
    F32,
    #[value(name = "float64")]
    F64,
}

impl From<DtypeArg> for Dtype {
    fn from(dtype: DtypeArg) -> Self {
        match dtype {
            DtypeArg::Bool => Dtype::Bool,
            DtypeArg::U8 => Dtype::U8,
            DtypeArg::U16 => Dtype::U16,
            DtypeArg::U32 => Dtype::U32,
            DtypeArg::U64 => Dtype::U64,
            DtypeArg::I8 => Dtype::I8,
            DtypeArg::I16 => Dtype::I16,
            DtypeArg::I32 => Dtype::I32,
            DtypeArg::I64 => Dtype::I64,
            DtypeArg::F32 => Dtype::F32,
            DtypeArg::F64 => Dtype::F64,
        }
    }
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

/// The d4 and bigWig imports take no format flags.
#[derive(Debug, Args)]
struct SignalArgs {
    #[command(flatten)]
    import: ImportOptions,
    #[command(flatten)]
    target: ImportTarget,
}

impl From<ImportModeArg> for ImportMode {
    fn from(mode: ImportModeArg) -> Self {
        match mode {
            ImportModeArg::Depth => ImportMode::Depth,
            ImportModeArg::Composition => ImportMode::Composition,
        }
    }
}

impl From<OverlapModeArg> for OverlapMode {
    fn from(overlap: OverlapModeArg) -> Self {
        match overlap {
            OverlapModeArg::Proper => OverlapMode::ProperOnly,
            OverlapModeArg::All => OverlapMode::All,
            OverlapModeArg::None => OverlapMode::None,
        }
    }
}

impl BamDecodeOptions {
    fn filter(&self) -> DepthFilter {
        DepthFilter {
            min_mapq: self.min_mapq,
            exclude_flags: self.exclude_flags,
            min_bq: self.min_bq,
            overlap: self.overlap.into(),
            count_deletions: self.count_deletions,
        }
    }
}

/// Raises the soft open-file limit to the hard limit. An import holds one
/// file handle per source at minimum, and macOS starts at a soft limit of 256.
#[cfg(unix)]
fn raise_fd_limit() {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit and setrlimit only read and write the given struct.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) != 0 || lim.rlim_cur >= lim.rlim_max {
            return;
        }
        let soft = lim.rlim_cur;
        lim.rlim_cur = lim.rlim_max;
        if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) != 0 {
            // macOS refuses a soft limit above OPEN_MAX (10240).
            lim.rlim_cur = lim.rlim_max.min(10240).max(soft);
            libc::setrlimit(libc::RLIMIT_NOFILE, &lim);
        }
    }
}

#[cfg(not(unix))]
fn raise_fd_limit() {}

fn main() -> Result<()> {
    raise_fd_limit();
    let cli = Cli::parse();
    init_logging(&cli.global);
    let quiet = cli.global.quiet;
    match cli.command {
        Command::View(args) => end_quietly_on_broken_pipe(view(args)),
        Command::Stat(args) => end_quietly_on_broken_pipe(stat(args)),
        Command::Import(import) => match import.format {
            ImportCommand::Bed(args) => import_bed(args, quiet),
            ImportCommand::Bam(args) => import_bam(args, quiet),
            ImportCommand::D4(args) => import_d4(args, quiet),
            ImportCommand::BigWig(args) => import_bigwig(args, quiet),
        },
        Command::Append(append) => match append.format {
            AppendCommand::Bed(args) => append_bed(args, quiet),
            AppendCommand::Bam(args) => append_bam(args, quiet),
            AppendCommand::D4(args) => append_d4(args, quiet),
            AppendCommand::BigWig(args) => append_bigwig(args, quiet),
        },
        Command::Info(args) => end_quietly_on_broken_pipe(info::run_info(&args.store, args.json)),
        Command::Validate(args) => {
            if !validate::run_validate(&args.store, args.json)? {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// A closed stdout, as from `pbz view | head`, is a normal end.
fn end_quietly_on_broken_pipe(result: Result<()>) -> Result<()> {
    match result {
        Err(err)
            if err.chain().any(|cause| {
                cause
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::BrokenPipe)
            }) =>
        {
            Ok(())
        }
        other => other,
    }
}

fn view(args: ViewArgs) -> Result<()> {
    view::run_view(&view::ViewSpec {
        store: args.store,
        track: args.track,
        regions: args.regions.region,
        regions_file: args.regions.regions,
        columns: args.columns,
        output: args.output,
        no_header: args.no_header,
        threads: args.threads,
        precision: args.precision,
    })
}

fn stat(args: StatArgs) -> Result<()> {
    stat::run_stat(&stat::StatSpec {
        store: args.store,
        track: args.track,
        regions: args.regions.region,
        regions_file: args.regions.regions,
        stat: args.stat.into(),
        columns: args.columns,
        output: args.output,
        no_header: args.no_header,
        threads: args.threads,
        precision: args.precision,
    })
}

fn import_bed(args: BedArgs, quiet: bool) -> Result<()> {
    let infer_rows = args
        .bed
        .infer_rows
        .as_deref()
        .map(parse_infer_rows)
        .transpose()?
        .unwrap_or_default();
    if let Some(bed) = &args.bed.emit_schema {
        info!("emitting a schema scaffold to stdout; no store is written");
        return emit_schema(bed, infer_rows);
    }

    let sources = args.bed.input.resolve_input("BED")?;
    debug!(
        "resolved {} BED source(s), infer rows {infer_rows:?}",
        sources.len()
    );

    let output = args.bed.store.context("STORE is required to import")?;
    let target = args.import.target(&output)?;
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
    let config = args.import.config(&label, quiet)?;
    let report = if let Some(schema_path) = &args.bed.schema {
        info!("loading BED schema {}", schema_path.display());
        let schema = load_schema(schema_path, &sources, infer_rows)?;
        let plan = plan_bed_schema(&sources, &schema, &config).context("plan BED schema")?;
        debug!("BED schema planned against {} source(s)", sources.len());
        execute_bed_schema_plan(target, plan, genome, config).context("import BED schema")?
    } else {
        let options = BedImportOptions {
            fields: (!args.bed.fields.is_empty()).then_some(args.bed.fields),
            track: args.bed.track,
            dtype: args.bed.dtype.map(Dtype::from),
            infer_rows,
            ..BedImportOptions::default()
        };
        from_bed_matrix(target, &sources, genome, &options, config).context("import BED columns")?
    };
    if !quiet {
        print_report("imported", sources.len(), &output, &report);
    }
    Ok(())
}

fn import_bam(args: BamArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("BAM/CRAM")?;
    let n_sources = sources.len();
    debug!("resolved {n_sources} alignment source(s)");

    let config = args.import.config(&args.target.track, quiet)?;
    let output = args.target.store;
    let report = from_bam(
        args.import.target(&output)?,
        &args.target.track,
        &sources,
        args.decode.mode.into(),
        args.decode.filter(),
        args.decode.fasta,
        config,
    )
    .context("import BAM/CRAM")?;

    if !quiet {
        print_report("imported", n_sources, &output, &report);
    }
    Ok(())
}

fn import_d4(args: SignalArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("d4")?;
    let n_sources = sources.len();
    debug!("resolved {n_sources} d4 source(s)");

    let config = args.import.config(&args.target.track, quiet)?;
    let output = args.target.store;
    let report = from_d4(
        args.import.target(&output)?,
        &args.target.track,
        &sources,
        config,
    )
    .context("import d4")?;

    if !quiet {
        print_report("imported", n_sources, &output, &report);
    }
    Ok(())
}

fn import_bigwig(args: SignalArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("bigWig")?;
    let n_sources = sources.len();
    debug!("resolved {n_sources} bigWig source(s)");

    let config = args.import.config(&args.target.track, quiet)?;
    let output = args.target.store;
    let report = from_bigwig(
        args.import.target(&output)?,
        &args.target.track,
        &sources,
        config,
    )
    .context("import bigWig")?;

    if !quiet {
        print_report("imported", n_sources, &output, &report);
    }
    Ok(())
}

/// Open the store an append grows and read the column dimension of `track`.
///
/// The dimension comes from the store so that one source still forms a column
/// batch; an append never names a new dimension.
fn open_append(path: &Path, track: &str) -> Result<(Dataset, String)> {
    if pbzarr::is_zip_store(path) || has_zip_extension(path) {
        bail!(
            "{} is a zip store. Zip stores are written once, so append cannot add columns to them",
            path.display()
        );
    }
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

fn print_report(verb: &str, sources: usize, path: &Path, report: &pbzarr::import::Report) {
    debug!(
        "wrote {} bytes in {:.1}s",
        report.bytes_written, report.wall_seconds
    );
    eprintln!(
        "{verb} {sources} sources -> {} ({} positions, {} chunks written, {} skipped)",
        path.display(),
        report.positions,
        report.chunks_written,
        report.chunks_skipped
    );
}

fn append_d4(args: AppendSignalArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("d4")?;
    let (mut dataset, dim) = open_append(&args.target.store, &args.target.track)?;
    let config = args.run.append_config(&args.target.track, dim, quiet);
    let report = from_d4(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        config,
    )
    .context("append d4")?;
    if !quiet {
        print_report("appended", sources.len(), &args.target.store, &report);
    }
    Ok(())
}

fn append_bigwig(args: AppendSignalArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("bigWig")?;
    let (mut dataset, dim) = open_append(&args.target.store, &args.target.track)?;
    let config = args.run.append_config(&args.target.track, dim, quiet);
    let report = from_bigwig(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        config,
    )
    .context("append bigWig")?;
    if !quiet {
        print_report("appended", sources.len(), &args.target.store, &report);
    }
    Ok(())
}

fn append_bam(args: AppendBamArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("BAM/CRAM")?;
    let (mut dataset, dim) = open_append(&args.target.store, &args.target.track)?;
    let config = args.run.append_config(&args.target.track, dim, quiet);
    let report = from_bam(
        Target::Append(&mut dataset),
        &args.target.track,
        &sources,
        args.decode.mode.into(),
        args.decode.filter(),
        args.decode.fasta,
        config,
    )
    .context("append BAM/CRAM")?;
    if !quiet {
        print_report("appended", sources.len(), &args.target.store, &report);
    }
    Ok(())
}

fn append_bed(args: AppendBedArgs, quiet: bool) -> Result<()> {
    let sources = args.target.input.resolve_input("BED")?;
    let (mut dataset, dim) = open_append(&args.target.store, &args.target.track)?;
    let genome = dataset.reference()?.clone();
    let dtype = Dtype::from_str(
        dataset.variables()[&args.target.track].metadata["data_type"]
            .as_str()
            .unwrap_or(""),
    )
    .with_context(|| format!("read the dtype of track {:?}", args.target.track))?;
    let config = args.run.append_config(&args.target.track, dim, quiet);
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
    if !quiet {
        print_report("appended", sources.len(), &args.target.store, &report);
    }
    Ok(())
}

fn read_schema_rows(text: &str, path: &Path) -> Result<Vec<BedColumnRequest>> {
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
        rows.push(BedColumnRequest {
            column: cells[0].to_owned(),
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

/// Build a `BedSchema` from a TSV schema file.
fn load_schema(schema_path: &Path, sources: &[Source], infer_rows: InferRows) -> Result<BedSchema> {
    let text = std::fs::read_to_string(schema_path)
        .with_context(|| format!("read schema {}", schema_path.display()))?;
    let rows = read_schema_rows(&text, schema_path)?;
    Ok(build_bed_schema(sources, &rows, infer_rows)?)
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
        assert_eq!(rows[0].column, "4");
        assert_eq!(rows[0].track.as_deref(), Some("depth"));
        assert_eq!(rows[0].dtype, Some(Dtype::I32));
        assert_eq!(rows[1].column, "mapq");
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
    fn store_is_first_and_track_is_a_flag() {
        let cli = Cli::try_parse_from([
            "pbz", "-v", "-q", "import", "d4", "out.pbz", "a.d4", "b.d4", "--track", "depth",
        ])
        .unwrap();
        assert_eq!(cli.global.verbose, 1);
        assert!(cli.global.quiet);
        let Command::Import(import) = cli.command else {
            panic!("expected import");
        };
        let ImportCommand::D4(args) = import.format else {
            panic!("expected import d4");
        };
        assert_eq!(args.target.store, PathBuf::from("out.pbz"));
        assert_eq!(args.target.input.input_files.len(), 2);
        assert_eq!(args.target.track, "depth");

        let cli = Cli::try_parse_from(["pbz", "view", "s.pbz", "--track", "af", "-c", "s1", "-v"])
            .unwrap();
        assert_eq!(cli.global.verbose, 1);
        let Command::View(view) = cli.command else {
            panic!("expected view");
        };
        assert_eq!(view.store, PathBuf::from("s.pbz"));
        assert_eq!(view.track.as_deref(), Some("af"));

        let err = Cli::try_parse_from([
            "pbz", "import", "d4", "-o", "out.pbz", "--track", "depth", "a.d4",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    #[test]
    fn sources_keep_urls_and_drop_compound_extensions() {
        let url = parse_source("https://h/x.bam").unwrap();
        assert_eq!(url.path, PathBuf::from("https://h/x.bam"));
        assert!(url.column_label.is_none());
        let labeled = parse_source("https://h/x.bam:S1").unwrap();
        assert_eq!(labeled.path, PathBuf::from("https://h/x.bam"));
        assert_eq!(labeled.label(), "S1");
        assert_eq!(parse_source("dir/s1.bed.gz").unwrap().label(), "s1");
    }

    #[test]
    fn exclude_flags_take_hex_and_flag_names() {
        let names = parse_sam_flags("UNMAP,SECONDARY,QCFAIL,DUP").unwrap();
        assert_eq!(parse_sam_flags("0x704").unwrap(), names);
        assert_eq!(parse_sam_flags("1796").unwrap(), names);
        assert!(parse_sam_flags("UNMAPPED").is_err());
    }

    #[test]
    fn codecs_conflicts_with_grid_flags() {
        let err = Cli::try_parse_from([
            "pbz",
            "import",
            "bed",
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
            let mut args = vec!["pbz", "import", "d4", "out.pbz", "--track", "depth"];
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
            let mut args = vec!["pbz", "import", "bam", "out.pbz", "--track", "depth"];
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
