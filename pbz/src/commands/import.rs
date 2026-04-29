use color_eyre::Result;
use color_eyre::eyre::eyre;

use crate::cli::ImportArgs;

pub fn run(_args: ImportArgs, _threads: Option<usize>, _no_progress: bool) -> Result<()> {
    Err(eyre!("not yet implemented"))
}
