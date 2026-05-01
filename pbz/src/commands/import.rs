use color_eyre::Result;
use color_eyre::eyre::eyre;
use pbzarr::{PbzStore, TrackConfig};
use std::collections::HashSet;
use std::io::BufRead;
use std::num::NonZero;
use std::path::Path;
use std::sync::Arc;

use crate::cli::ImportArgs;
use crate::commands::input_resolve::{
    InputFormat, InputSpec, check_unique, parse_filelist, parse_input_spec,
};
use crate::io::bed_reader::{BedFlavor, BedIntervalSource, BedPerBaseReader};
use crate::io::d4_reader::D4Reader;
use crate::io::value_reader::{ValueDtype, ValueReader};
use crate::limits::check_fd_budget;
use crate::pipeline::ImportPipeline;
use crate::progress::maybe_bar;

pub fn run(args: ImportArgs, threads: Option<usize>, no_progress: bool) -> Result<()> {
    if args.chunk_size == 0 {
        return Err(eyre!("--chunk-size must be greater than 0"));
    }
    if args.sample_chunk_size == 0 {
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

    // Forbid mixing input formats within one track.
    let formats: HashSet<&InputFormat> = specs.iter().map(|s| &s.format).collect();
    if formats.len() > 1 {
        return Err(eyre!(
            "input formats must match within a single track; got: {:?}",
            formats
        ));
    }
    let any_bed = specs.iter().any(|s| s.format == InputFormat::Bed);

    let n_threads = threads.unwrap_or_else(num_cpus::get).max(1);
    check_fd_budget(specs.len(), n_threads)?;

    // --contigs validation: required iff (new store AND any BED), rejected otherwise.
    let store_exists = args.store.exists();
    match (store_exists, any_bed, args.contigs.is_some()) {
        (true, _, true) => {
            return Err(eyre!(
                "--contigs is rejected when STORE already exists; contigs come from the store"
            ));
        }
        (false, false, true) => {
            return Err(eyre!(
                "--contigs is rejected when all inputs are D4; D4 carries its own contig header"
            ));
        }
        (false, true, false) => {
            return Err(eyre!(
                "--contigs path.fai is required when creating a new store from BED inputs"
            ));
        }
        _ => {}
    }

    // Resolve contigs needed by BED readers (and used to seed a new store from BED).
    // For D4-only paths we leave this empty and rely on per-reader contigs as before.
    let resolved_contigs: Vec<(String, u64)> = if any_bed {
        if store_exists {
            let s = PbzStore::open(&args.store)?;
            let names = s.contigs().to_vec();
            let lens = s.contig_lengths();
            let mut out = Vec::with_capacity(names.len());
            for n in &names {
                let l = *lens.get(n).ok_or_else(|| {
                    eyre!("store missing length for contig {n}")
                })?;
                out.push((n.clone(), l));
            }
            out
        } else {
            let p = args.contigs.as_ref().expect("validated above");
            parse_contigs_file(p)?
        }
    } else {
        Vec::new()
    };

    let user_dtype = args.dtype.as_deref();
    let worker_count = NonZero::new(n_threads.min(4).max(1)).unwrap();

    let mut bed_flavors: Vec<BedFlavor> = Vec::new();
    let mut readers: Vec<Box<dyn ValueReader>> = Vec::with_capacity(specs.len());
    for s in &specs {
        match s.format {
            InputFormat::D4 => {
                let r = D4Reader::open(&s.path)?;
                readers.push(Box::new(r));
            }
            InputFormat::Bed => {
                let src = BedIntervalSource::open(
                    &s.path,
                    resolved_contigs.clone(),
                    worker_count,
                )?;
                let flavor = src.flavor;
                bed_flavors.push(flavor);
                let dtype = match (flavor, user_dtype) {
                    (BedFlavor::Mask, Some(_)) => {
                        return Err(eyre!(
                            "--dtype is only valid for bedGraph inputs; {} is BED3 (mask)",
                            s.path.display()
                        ));
                    }
                    (BedFlavor::Mask, None) => ValueDtype::Bool,
                    (BedFlavor::BedGraph, None) => ValueDtype::F32,
                    (BedFlavor::BedGraph, Some(d)) => parse_dtype_str(d)?,
                };
                let r = BedPerBaseReader::from_source(src, dtype)?;
                readers.push(Box::new(r));
            }
        }
    }

    // Cross-reader BED flavor consistency.
    if let Some((first, rest)) = bed_flavors.split_first() {
        for f in rest {
            if f != first {
                return Err(eyre!(
                    "all BED inputs to one track must share the same flavor; got {first:?} and {f:?}"
                ));
            }
        }
    }

    let first_dtype = readers[0].dtype();
    for r in readers.iter().skip(1) {
        if r.dtype() != first_dtype {
            let pairs: Vec<(String, ValueDtype)> = specs
                .iter()
                .zip(readers.iter())
                .map(|(s, r)| (s.path.display().to_string(), r.dtype()))
                .collect();
            return Err(eyre!("input dtypes disagree: {:?}", pairs));
        }
    }
    let dtype: ValueDtype = match (any_bed, user_dtype) {
        // For BED, dtype was already resolved per-reader; user_dtype was either
        // consumed (bedGraph) or rejected (mask). Don't re-apply at the track level.
        (true, _) => first_dtype,
        (false, Some(d)) => parse_dtype_str(d)?,
        (false, None) => first_dtype,
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

    // v1: every sample is one input. With a single input we use a 1D scalar
    // track (per spec: single-sample data MUST be 1D); otherwise multi-sample 2D.
    let sample_names: Vec<String> = specs.iter().map(|s| s.column_name.clone()).collect();
    let has_samples = sample_names.len() > 1;
    let cfg = TrackConfig {
        dtype: dtype.as_str().to_string(),
        samples: if has_samples { Some(sample_names) } else { None },
        chunk_size: args.chunk_size,
        sample_chunk_size: args.sample_chunk_size,
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
        has_samples,
    };
    pipeline.run()?;
    Ok(())
}

fn parse_dtype_str(s: &str) -> Result<ValueDtype> {
    match s {
        "bool" => Ok(ValueDtype::Bool),
        "uint8" => Ok(ValueDtype::U8),
        "uint16" => Ok(ValueDtype::U16),
        "uint32" => Ok(ValueDtype::U32),
        "int8" => Ok(ValueDtype::I8),
        "int16" => Ok(ValueDtype::I16),
        "int32" => Ok(ValueDtype::I32),
        "float32" => Ok(ValueDtype::F32),
        "float64" => Ok(ValueDtype::F64),
        other => Err(eyre!("unrecognized --dtype {other:?}")),
    }
}

pub(crate) fn parse_contigs_file(path: &Path) -> Result<Vec<(String, u64)>> {
    let f = std::fs::File::open(path)
        .map_err(|e| eyre!("failed to open --contigs {}: {e}", path.display()))?;
    let r = std::io::BufReader::new(f);
    let mut out: Vec<(String, u64)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (lineno, line) in r.lines().enumerate() {
        let line = line.map_err(|e| eyre!("error reading {}:{}: {e}", path.display(), lineno + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut it = trimmed.split('\t');
        let name = it.next().ok_or_else(|| eyre!("malformed line {}:{}", path.display(), lineno + 1))?;
        let length_s = it.next().ok_or_else(|| {
            eyre!(
                "malformed line {}:{}: expected at least 2 tab-separated columns",
                path.display(),
                lineno + 1
            )
        })?;
        let length: u64 = length_s.parse().map_err(|_| {
            eyre!(
                "malformed line {}:{}: length '{length_s}' is not a non-negative integer",
                path.display(),
                lineno + 1
            )
        })?;
        if !seen.insert(name.to_string()) {
            return Err(eyre!(
                "duplicate contig '{name}' at {}:{}",
                path.display(),
                lineno + 1
            ));
        }
        out.push((name.to_string(), length));
    }
    if out.is_empty() {
        return Err(eyre!("--contigs file {} is empty (no contigs)", path.display()));
    }
    Ok(out)
}

#[cfg(test)]
mod fai_tests {
    use super::*;

    fn write_fai(text: &str) -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new()
            .prefix("contigs-")
            .suffix(".fai")
            .tempfile()
            .unwrap();
        std::fs::write(f.path(), text).unwrap();
        f
    }

    #[test]
    fn parses_chrom_sizes_two_columns() {
        let f = write_fai("chr1\t1000000\nchr2\t500000\n");
        let v = parse_contigs_file(f.path()).unwrap();
        assert_eq!(v, vec![
            ("chr1".to_string(), 1_000_000),
            ("chr2".to_string(), 500_000),
        ]);
    }

    #[test]
    fn parses_fai_extra_columns_ignored() {
        let f = write_fai("chr1\t1000000\t52\t60\t61\nchr2\t500000\t1052689\t60\t61\n");
        let v = parse_contigs_file(f.path()).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], ("chr1".to_string(), 1_000_000));
    }

    #[test]
    fn rejects_empty() {
        let f = write_fai("");
        let err = parse_contigs_file(f.path()).unwrap_err();
        assert!(err.to_string().contains("empty") || err.to_string().contains("no contigs"));
    }

    #[test]
    fn rejects_non_numeric_length() {
        let f = write_fai("chr1\tnot_a_number\n");
        let err = parse_contigs_file(f.path()).unwrap_err();
        assert!(err.to_string().contains("length"));
    }

    #[test]
    fn rejects_duplicate_contig() {
        let f = write_fai("chr1\t100\nchr1\t200\n");
        let err = parse_contigs_file(f.path()).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }
}
