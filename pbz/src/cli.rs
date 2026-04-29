use clap::{Parser, Subcommand, ValueEnum};
use color_eyre::Result;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "pbz", version, about = "PBZ (Per-Base Zarr) command-line tool")]
pub struct Cli {
    /// Increase log verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Disable progress bars (also auto-disabled when stderr is not a TTY)
    #[arg(long, global = true)]
    pub no_progress: bool,

    /// Number of worker threads (default: number of CPUs)
    #[arg(long, global = true)]
    pub threads: Option<usize>,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        crate::commands::init_tracing(self.verbose);
        match self.command {
            Command::Import(a) => crate::commands::import::run(a, self.threads, self.no_progress),
            Command::Export(a) => crate::commands::export::run(a),
            Command::Cat(a) => crate::commands::cat::run(a),
            Command::Info(a) => crate::commands::info::run(a),
            Command::List(a) => crate::commands::list::run(a),
            Command::Set(a) => crate::commands::set::run(a),
            Command::Rename(a) => crate::commands::rename::run(a),
            Command::Drop(a) => crate::commands::drop_cmd::run(a),
            Command::Validate(a) => crate::commands::validate::run(a),
            Command::Completions(a) => crate::commands::completions::run(a),
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Import data into a track (creating the store/track if needed).
    Import(ImportArgs),
    /// Export a track to bedGraph/BED/TSV.
    Export(ExportArgs),
    /// Print a track to stdout.
    Cat(CatArgs),
    /// Show store and track metadata.
    Info(InfoArgs),
    /// List tracks, contigs, or columns.
    List(ListArgs),
    /// Edit a track's description or source metadata.
    Set(SetArgs),
    /// Rename a track.
    Rename(RenameArgs),
    /// Delete a track.
    Drop(DropArgs),
    /// Validate a store against the spec.
    Validate(ValidateArgs),
    /// Print shell completion scripts.
    Completions(CompletionsArgs),
}

#[derive(clap::Args, Debug)]
#[command(group(
    clap::ArgGroup::new("import_input").required(true).args(["input", "filelist"])
))]
pub struct ImportArgs {
    /// Path to the .pbz.zarr store.
    pub store: PathBuf,
    /// Track name to create.
    #[arg(long)]
    pub track: String,
    /// Input file with optional :COLUMN_NAME suffix. Repeatable.
    #[arg(short = 'i', long = "input")]
    pub input: Vec<String>,
    /// File listing one input per line (path or path:column_name).
    #[arg(short = 'f', long)]
    pub filelist: Option<PathBuf>,
    /// Position-axis chunk size.
    #[arg(long, default_value_t = 1_000_000)]
    pub chunk_size: u64,
    /// Column-axis chunk size.
    #[arg(long, default_value_t = 16)]
    pub column_chunk_size: u64,
    /// Track description.
    #[arg(long)]
    pub description: Option<String>,
    /// Track source string.
    #[arg(long)]
    pub source: Option<String>,
    /// Override the inferred dtype.
    #[arg(long)]
    pub dtype: Option<String>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ExportFormat {
    Auto,
    Bedgraph,
    Bed,
    Tsv,
}

#[derive(clap::Args, Debug)]
pub struct ExportArgs {
    pub store: PathBuf,
    pub track: String,
    /// Region in `chr1`, `chr1:1000-2000`, or `chr1:1000` form (0-based half-open).
    #[arg(long)]
    pub region: Option<String>,
    /// Filter to specific column names (TSV only). Repeatable.
    #[arg(long)]
    pub column: Vec<String>,
    /// Output file path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Output format. `auto` infers from the output extension.
    #[arg(long, value_enum, default_value_t = ExportFormat::Auto)]
    pub format: ExportFormat,
    /// Include runs of fill_value (e.g. zero) in bedGraph output.
    #[arg(long)]
    pub include_zero: bool,
}

#[derive(clap::Args, Debug)]
pub struct CatArgs {
    pub store: PathBuf,
    pub track: String,
    #[arg(long)]
    pub region: Option<String>,
    #[arg(long)]
    pub column: Vec<String>,
    /// Format (default: tsv).
    #[arg(long, value_enum, default_value_t = ExportFormat::Tsv)]
    pub format: ExportFormat,
    #[arg(long)]
    pub include_zero: bool,
}

#[derive(clap::Args, Debug)]
pub struct InfoArgs {
    pub store: PathBuf,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
#[command(group(
    clap::ArgGroup::new("list_mode").required(true).args(["tracks", "contigs", "columns"])
))]
pub struct ListArgs {
    pub store: PathBuf,
    #[arg(long)]
    pub tracks: bool,
    #[arg(long)]
    pub contigs: bool,
    #[arg(long, value_name = "TRACK")]
    pub columns: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
#[command(group(
    clap::ArgGroup::new("set_field").required(true).multiple(true).args(["description", "source"])
))]
pub struct SetArgs {
    pub store: PathBuf,
    pub track: String,
    /// New description (empty string clears).
    #[arg(long)]
    pub description: Option<String>,
    /// New source (empty string clears).
    #[arg(long)]
    pub source: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct RenameArgs {
    pub store: PathBuf,
    pub old_name: String,
    pub new_name: String,
}

#[derive(clap::Args, Debug)]
pub struct DropArgs {
    pub store: PathBuf,
    pub track: String,
    /// Skip interactive confirmation.
    #[arg(long)]
    pub yes: bool,
}

#[derive(clap::Args, Debug)]
pub struct ValidateArgs {
    pub store: PathBuf,
    #[arg(long)]
    pub json: bool,
}

#[derive(clap::Args, Debug)]
pub struct CompletionsArgs {
    #[arg(value_enum)]
    pub shell: clap_complete::Shell,
}
