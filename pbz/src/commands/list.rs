use color_eyre::Result;
use color_eyre::eyre::eyre;
use pbzarr::PbzStore;

use crate::cli::ListArgs;

pub fn run(args: ListArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let items: Vec<String> = if args.tracks {
        store.tracks()?
    } else if args.contigs {
        store.contigs().to_vec()
    } else if let Some(track_name) = args.samples {
        let track = store.track(&track_name)?;
        track
            .samples()
            .map(|c| c.to_vec())
            .ok_or_else(|| eyre!("track '{track_name}' is scalar; no samples to list"))?
    } else {
        unreachable!("clap ArgGroup ensures one mode flag is set")
    };

    if args.json {
        println!("{}", serde_json::to_string(&items)?);
    } else {
        for item in items {
            println!("{item}");
        }
    }
    Ok(())
}
