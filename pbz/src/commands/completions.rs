use clap::CommandFactory;
use color_eyre::Result;

use crate::cli::{Cli, CompletionsArgs};

pub fn run(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = "pbz".to_string();
    clap_complete::generate(args.shell, &mut cmd, bin_name, &mut std::io::stdout());
    Ok(())
}
