use color_eyre::Result;
use pbzarr::PbzStore;

use crate::cli::RenameArgs;

pub fn run(args: RenameArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    store.rename_track(&args.old_name, &args.new_name)?;
    Ok(())
}
