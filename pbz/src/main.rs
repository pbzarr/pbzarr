mod cli;
mod commands;
mod error;
mod io;
mod limits;
mod pipeline;
mod progress;

use clap::Parser;
use color_eyre::Result;

fn main() -> Result<()> {
    error::install()?;
    let cli = cli::Cli::parse();
    cli.run()
}
