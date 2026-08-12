use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use clap::{ArgAction, Args, Parser, Subcommand};
use pbzarr::import::Config;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore};
use pbzarr_readers::{BedImportOptions, BedSource, InferRows, from_bed_matrix};

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
    Import(ImportArgs),
}

#[derive(Debug, Args)]
struct ViewArgs {
    #[command(flatten)]
    global: GlobalOptions,
    /// PBZ store to inspect.
    store: PathBuf,
    /// Track to view (defaults to first track if multiple present)
    track: Option<String>
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
}

#[derive(Debug, Args)]
#[command(
    arg_required_else_help = true,
    after_help = "Examples:\n  pbz import bed --output coverage.pbz --genome genome.fai coverage.bed.gz:sample-a\n  pbz import bed --output coverage.pbz --genome genome.fai --file-list sources.tsv"
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
    /// Increase logging verbosity. Repeated uses are accepted.
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
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
    /// Name for the column dimension of a multi-source track.
    #[arg(long)]
    column_dim: Option<String>,
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
    #[arg(short('o'), long)]
    output: PathBuf,
    #[command(flatten)]
    input: ImportInputOptions,
    /// FAI or chromosome-sizes file defining the genome.
    #[arg(long)]
    genome: PathBuf,
    /// BED value field to import. Repeating selects several fields; omit to import all fields.
    #[arg(long = "field")]
    fields: Vec<String>,
    /// Override a field's inferred dtype as FIELD=DTYPE.
    #[arg(long = "dtype", value_name = "FIELD=DTYPE")]
    dtype_overrides: Vec<String>,
    /// Records sampled for dtype inference, or `all`.
    #[arg(long, default_value = "1000", value_name = "N|all")]
    infer_rows: String,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::View(args) => view(args),
        Command::Import(ImportArgs {
            format: ImportCommand::Bed(args),
        }) => import_bed(args),
    }
}

fn view(args: ViewArgs) -> Result<()> {
    let _ = args.global.verbose;
    bail!("view is not implemented yet for {}", args.store.display())
}

fn import_bed(args: BedArgs) -> Result<()> {
    // TODO: setup logging/tracing
    let _ = args.global.verbose;
    let genome = Genome::from_fai(&args.bed.genome)
        .with_context(|| format!("read genome {}", args.bed.genome.display()))?;
    let sources: Vec<BedSource> = args
        .bed
        .input
        .resolve_input()?
        .into_iter()
        .map(|source| BedSource {
            path: source.path,
            column_label: source.label,
        })
        .collect();
    let options = BedImportOptions {
        fields: (!args.bed.fields.is_empty()).then_some(args.bed.fields),
        dtype_overrides: parse_dtype_overrides(args.bed.dtype_overrides)?,
        infer_rows: parse_infer_rows(&args.bed.infer_rows)?,
    };
    let config = Config {
        workers: args.import.threads,
        chunk_size: args.import.chunk_size,
        column_chunk_size: args.import.column_chunk_size,
        shard_size: args.import.shard_size,
        shard_column_size: args.import.shard_column_size,
        column_dim: args.import.column_dim,
        ..Config::default()
    };
    let mut store = if args.bed.output.exists() {
        PbzStore::open(&args.bed.output)
            .with_context(|| format!("open output store {}", args.bed.output.display()))?
    } else {
        PbzStore::create(&args.bed.output)
            .with_context(|| format!("create output store {}", args.bed.output.display()))?
    };

    from_bed_matrix(&mut store, &sources, genome, &options, config).context("import BED fields")?;
    Ok(())
}



fn parse_dtype_overrides(values: Vec<String>) -> Result<BTreeMap<String, Dtype>> {
    let mut overrides = BTreeMap::new();
    for value in values {
        let (field, dtype) = value
            .split_once('=')
            .filter(|(field, dtype)| !field.is_empty() && !dtype.is_empty())
            .with_context(|| format!("invalid dtype override {value:?}; expected FIELD=DTYPE"))?;
        let dtype = Dtype::from_str(dtype)
            .with_context(|| format!("invalid dtype override for field {field:?}"))?;
        if overrides.insert(field.to_owned(), dtype).is_some() {
            bail!("dtype override for field {field:?} was supplied more than once");
        }
    }
    Ok(overrides)
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
