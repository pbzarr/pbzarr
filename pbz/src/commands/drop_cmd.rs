use color_eyre::Result;
use color_eyre::eyre::eyre;
use pbzarr::PbzStore;
use std::io::IsTerminal;

use crate::cli::DropArgs;

pub fn run(args: DropArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let track = store.track(&args.track)?;
    let m = track.metadata();
    let dtype = m.dtype.clone();
    let cols_msg = if track.has_samples() {
        format!(
            ", {} samples",
            track.samples().map(|c| c.len()).unwrap_or(0)
        )
    } else {
        ", scalar".to_string()
    };
    drop(track);

    if !args.yes {
        if !std::io::stdin().is_terminal() {
            return Err(eyre!(
                "--yes required for non-interactive use (stdin is not a TTY)"
            ));
        }
        eprintln!(
            "about to drop track '{}' ({}{}). Proceed? [y/N]: ",
            args.track, dtype, cols_msg
        );
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf)?;
        let confirmed = matches!(buf.trim().to_ascii_lowercase().as_str(), "y" | "yes");
        if !confirmed {
            eprintln!("aborted");
            return Ok(());
        }
    }
    store.drop_track(&args.track)?;
    Ok(())
}
