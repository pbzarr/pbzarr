use color_eyre::Result;
use pbzarr::PbzStore;
use std::fs::File;
use std::io::BufWriter;

use crate::cli::ExportArgs;
use crate::commands::export_engine::{build_plan, make_writer, run_export};
use crate::io::pick_format;

pub fn run(args: ExportArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let track = store.track(&args.track)?;
    let fmt = pick_format(args.format, Some(&args.output), &track)?;
    let plan = build_plan(&store, &track, args.region.as_deref(), &args.column)?;
    let f = BufWriter::new(File::create(&args.output)?);
    let writer = make_writer(fmt, f, &track, args.include_zero)?;
    run_export(&track, &plan, writer)
}
