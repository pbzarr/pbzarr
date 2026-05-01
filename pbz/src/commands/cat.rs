use color_eyre::Result;
use pbzarr::PbzStore;
use std::io::{self, BufWriter};

use crate::cli::{CatArgs, ExportFormat};
use crate::commands::export_engine::{build_plan, make_writer, run_export};
use crate::io::pick_format;

pub fn run(args: CatArgs) -> Result<()> {
    let store = PbzStore::open(&args.store)?;
    let track = store.track(&args.track)?;
    // For cat, ExportFormat::Auto becomes Tsv (no output file to infer from).
    let effective_format = match args.format {
        ExportFormat::Auto => ExportFormat::Tsv,
        f => f,
    };
    let fmt = pick_format(effective_format, None, &track)?;
    let plan = build_plan(&store, &track, args.region.as_deref(), &args.sample)?;
    let stdout = io::stdout();
    let writer = make_writer(
        fmt,
        BufWriter::new(stdout.lock()),
        &track,
        args.include_zero,
    )?;
    run_export(&track, &plan, writer)
}
