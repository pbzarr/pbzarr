use color_eyre::Result;
use pbzarr::PbzStore;

use crate::cli::SetArgs;

pub fn run(args: SetArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let mut track = store.track(&args.track)?;
    if let Some(d) = &args.description {
        let val = if d.is_empty() { None } else { Some(d.as_str()) };
        track.set_description(val)?;
    }
    if let Some(s) = &args.source {
        let val = if s.is_empty() { None } else { Some(s.as_str()) };
        track.set_source(val)?;
    }
    Ok(())
}
