use color_eyre::Result;
use color_eyre::eyre::eyre;
use pbzarr::{PbzStore, TrackConfig};
use std::sync::Arc;

use crate::cli::ImportArgs;
use crate::commands::input_resolve::{
    InputFormat, InputSpec, check_unique, parse_filelist, parse_input_spec,
};
use crate::io::d4_reader::D4Reader;
use crate::io::depth_reader::DepthReader;
use crate::limits::check_fd_budget;
use crate::pipeline::ImportPipeline;
use crate::progress::maybe_bar;

pub fn run(args: ImportArgs, threads: Option<usize>, no_progress: bool) -> Result<()> {
    if args.chunk_size == 0 {
        return Err(eyre!("--chunk-size must be greater than 0"));
    }
    if args.column_chunk_size == 0 {
        return Err(eyre!("--column-chunk-size must be greater than 0"));
    }

    let specs: Vec<InputSpec> = if let Some(filelist) = &args.filelist {
        parse_filelist(filelist)?
    } else {
        let mut v = Vec::with_capacity(args.input.len());
        for s in &args.input {
            v.push(parse_input_spec(s)?);
        }
        v
    };
    check_unique(&specs)?;
    if specs.is_empty() {
        return Err(eyre!("no inputs provided"));
    }

    let n_threads = threads.unwrap_or_else(num_cpus::get).max(1);
    check_fd_budget(specs.len(), n_threads)?;

    let readers: Vec<Box<dyn DepthReader>> = specs
        .iter()
        .map(|s| match s.format {
            InputFormat::D4 => {
                let r = D4Reader::open(&s.path)?;
                Ok(Box::new(r) as Box<dyn DepthReader>)
            }
        })
        .collect::<Result<_>>()?;

    let dtype = match args.dtype.clone() {
        Some(d) => d,
        None => {
            let dts: Vec<(String, &str)> = specs
                .iter()
                .zip(readers.iter())
                .map(|(s, r)| (s.path.display().to_string(), r.dtype()))
                .collect();
            if dts.iter().any(|(_, d)| *d != dts[0].1) {
                return Err(eyre!("input dtypes disagree: {:?}", dts));
            }
            dts[0].1.to_string()
        }
    };

    let first_contigs = readers[0].contigs().to_vec();
    for (i, r) in readers.iter().enumerate().skip(1) {
        if r.contigs() != first_contigs.as_slice() {
            return Err(eyre!(
                "input '{}' has different contigs than '{}'",
                specs[i].path.display(),
                specs[0].path.display(),
            ));
        }
    }

    let store = if args.store.exists() {
        let s = PbzStore::open(&args.store)?;
        for (name, len) in &first_contigs {
            let stored = s.contig_length(name)?;
            if stored != *len {
                return Err(eyre!(
                    "contig {name}: store has length {stored}, input reports {len}"
                ));
            }
        }
        s
    } else {
        let names: Vec<String> = first_contigs.iter().map(|(n, _)| n.clone()).collect();
        let lens: Vec<u64> = first_contigs.iter().map(|(_, l)| *l).collect();
        PbzStore::create(&args.store, &names, &lens)?
    };

    // v1: every column is one input; track is always columnar.
    let cols: Vec<String> = specs.iter().map(|s| s.column_name.clone()).collect();
    let cfg = TrackConfig {
        dtype: dtype.clone(),
        columns: Some(cols),
        chunk_size: args.chunk_size,
        column_chunk_size: args.column_chunk_size,
        description: args.description.clone(),
        source: args.source.clone(),
        extra: serde_json::Map::new(),
    };
    let track = Arc::new(store.create_track(&args.track, cfg)?);

    let total_chunks: u64 = first_contigs
        .iter()
        .map(|(_, l)| l.div_ceil(args.chunk_size))
        .sum();
    let progress = maybe_bar(total_chunks, no_progress);

    let pipeline = ImportPipeline {
        readers,
        track,
        contigs: first_contigs,
        chunk_size: args.chunk_size,
        reader_workers: n_threads,
        writer_workers: n_threads,
        progress,
        dtype,
        has_columns: true,
    };
    pipeline.run()?;
    Ok(())
}
