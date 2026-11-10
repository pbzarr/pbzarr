//! `pbz stat`: summary statistics for one track.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use noodles_bgzf as bgzf;
use pbzarr::stat::{StatKind, StatOptions, StatOutput, StatResult, StatValue};
use pbzarr::{Region, RegionQuery};

use crate::fmt::write_g;
use crate::regions::{NamedRegion, select_regions};

pub(crate) struct StatSpec {
    pub store: PathBuf,
    pub track: Option<String>,
    pub regions: Vec<RegionQuery>,
    pub regions_file: Option<PathBuf>,
    pub stat: String,
    pub columns: Vec<String>,
    pub output: Option<PathBuf>,
    pub no_header: bool,
    pub threads: Option<NonZeroUsize>,
    pub precision: u8,
}

pub(crate) fn run_stat(spec: &StatSpec) -> Result<()> {
    if !(1..=17).contains(&spec.precision) {
        bail!("--precision must be between 1 and 17 significant digits");
    }
    let kind: StatKind = spec.stat.parse().map_err(anyhow::Error::new)?;
    // One rayon pool serves zarrs chunk decode, stat batches, and BGZF.
    if let Some(threads) = spec.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.get())
            .build_global()
            .context("configure worker thread pool")?;
    }
    let store = crate::view::open_store(&spec.store)?;
    let measurement = crate::view::resolve_measurement(&store, spec.track.as_deref())?;
    let named = select_regions(
        measurement.reference(),
        &spec.regions,
        spec.regions_file.as_ref(),
    )?;
    let regions: Vec<Region> = named.iter().map(|n| n.region).collect();
    let options = StatOptions {
        columns: (!spec.columns.is_empty()).then(|| spec.columns.clone()),
    };
    let result = pbzarr::stat::run(&measurement, &regions, kind, &options)?;
    let precision = usize::from(spec.precision);
    match spec.output.as_deref() {
        None => {
            let mut out = BufWriter::new(io::stdout());
            write_output(&named, kind, &result, spec.no_header, precision, &mut out)?;
            out.flush()?;
        }
        Some(path)
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gz")) =>
        {
            let file =
                File::create(path).with_context(|| format!("create output {}", path.display()))?;
            let mut out = bgzf::io::MultithreadedWriter::new(file);
            write_output(&named, kind, &result, spec.no_header, precision, &mut out)?;
            out.finish()?;
        }
        Some(path) => {
            let file =
                File::create(path).with_context(|| format!("create output {}", path.display()))?;
            let mut out = BufWriter::new(file);
            write_output(&named, kind, &result, spec.no_header, precision, &mut out)?;
            out.flush()?;
        }
    }
    Ok(())
}

fn write_output<W: Write>(
    named: &[NamedRegion],
    kind: StatKind,
    output: &StatOutput,
    no_header: bool,
    precision: usize,
    out: &mut W,
) -> Result<()> {
    match &output.result {
        StatResult::PerRegion(rows) => {
            if !no_header {
                write!(out, "#chrom\tstart\tend")?;
                if output.samples.is_empty() {
                    write!(out, "\t{}", kind.name())?;
                } else {
                    for sample in &output.samples {
                        write!(out, "\t{sample}")?;
                    }
                }
                writeln!(out)?;
            }
            debug_assert_eq!(named.len(), rows.len());
            for (region, row) in named.iter().zip(rows) {
                write!(
                    out,
                    "{}\t{}\t{}",
                    region.chrom, region.region.start, region.region.end
                )?;
                for value in row {
                    out.write_all(b"\t")?;
                    write_value(out, *value, precision)?;
                }
                writeln!(out)?;
            }
        }
        StatResult::Hist(table) => {
            if !no_header {
                write!(out, "#value")?;
                if output.samples.is_empty() {
                    write!(out, "\tcount")?;
                } else {
                    for sample in &output.samples {
                        write!(out, "\t{sample}")?;
                    }
                }
                writeln!(out)?;
            }
            for (index, value) in table.values.iter().enumerate() {
                write!(out, "{value}")?;
                for count in &table.counts[index] {
                    write!(out, "\t{count}")?;
                }
                writeln!(out)?;
            }
        }
    }
    Ok(())
}

fn write_value<W: Write>(out: &mut W, value: StatValue, precision: usize) -> io::Result<()> {
    match value {
        StatValue::Int(v) => write!(out, "{v}"),
        StatValue::Float(v) => write_g(out, v, precision),
    }
}
