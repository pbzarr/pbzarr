//! Usage:
//!   cargo build --release --example profile_import
//!   ./target/release/examples/profile_import \
//!       --contigs 4 --length 50000000 --cols 8 \
//!       --chunk 65536 --shard 1024 --workers 4
//!
//! Profile with samply:
//!   pixi global install samply  (or: cargo install samply)
//!   samply record -- ./target/release/examples/profile_import --contigs 2 --length 100000000

use std::path::PathBuf;
use std::time::Instant;

use pbzarr::import::{Config, Import, Source};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, TrackConfig};
use pbzarr::{Dataset, Reference};
use pbzarr_readers::from_d4;
use tempfile::TempDir;

#[derive(Debug)]
struct Args {
    contigs: usize,
    length: u64,
    cols: usize,
    chunk: usize,
    /// `None` (from `--unsharded`) writes one object per inner chunk.
    shard: Option<usize>,
    shard_columns: usize,
    workers: usize,
    keep: bool,
    out: Option<PathBuf>,
    /// If set, import using `D4Reader` against this file (replicated `cols` times).
    /// Reference and `length` come from the d4 header.
    d4: Option<PathBuf>,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            contigs: 4,
            length: 50_000_000,
            cols: 8,
            chunk: pbzarr::DEFAULT_CHUNK,
            shard: Some(pbzarr::DEFAULT_SHARD),
            shard_columns: pbzarr::DEFAULT_SHARD_COLUMNS,
            workers: 4,
            keep: false,
            out: None,
            d4: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--contigs" => a.contigs = it.next().unwrap().parse().unwrap(),
                "--length" => a.length = it.next().unwrap().parse().unwrap(),
                "--cols" => a.cols = it.next().unwrap().parse().unwrap(),
                "--chunk" => a.chunk = it.next().unwrap().parse().unwrap(),
                "--shard" => a.shard = Some(it.next().unwrap().parse().unwrap()),
                "--shard-columns" => a.shard_columns = it.next().unwrap().parse().unwrap(),
                "--unsharded" => a.shard = None,
                "--workers" => a.workers = it.next().unwrap().parse().unwrap(),
                "--keep" => a.keep = true,
                "--out" => a.out = Some(PathBuf::from(it.next().unwrap())),
                "--d4" => a.d4 = Some(PathBuf::from(it.next().unwrap())),
                "-h" | "--help" => {
                    eprintln!(
                        "usage: profile_import [--contigs N] [--length L] [--cols C] \
                         [--chunk N] [--shard N] [--shard-columns N] [--unsharded] \
                         [--workers W] [--keep] [--out PATH]"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown arg: {other}"),
            }
        }
        a
    }
}

/// Synthetic reader that fills buffers with a deterministic pattern.
/// Pattern defeats trivial compression but is cheap to compute, so the cost
/// stays on the writer (encode + IO) rather than the reader.
struct SynthReader {
    genome: Reference,
    schema: OutputSchema,
    seed: u32,
}

impl SynthReader {
    fn new(genome: Reference, seed: u32) -> Self {
        Self {
            genome,
            schema: OutputSchema::single("depth", Dtype::U32),
            seed,
        }
    }
}

impl ValueReader for SynthReader {
    fn contigs(&self) -> &Reference {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        _contig: &str,
        start: u64,
        end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        let len = (end - start) as usize;
        let s = self.seed;
        let dst = outputs[0].as_u32_mut()?;
        for i in 0..len {
            // Cheap LCG-ish mix. Blosc shuffle + zstd still compress some.
            let v = ((start as u32)
                .wrapping_add(i as u32)
                .wrapping_mul(2654435761u32))
                ^ s;
            dst[i] = v & 0x3FF; // depth-like: 0..1024
        }
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(Self::new(self.genome.clone(), self.seed))
    }
}

fn human_rate(bytes: u64, secs: f64) -> String {
    let mbs = (bytes as f64) / 1_048_576.0 / secs;
    format!("{mbs:.1} MiB/s")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("config: {args:?}");

    // For --d4 mode, override the synthetic contigs with the d4 file's genome.
    let (genome, total_length): (Reference, u64) = if let Some(ref d4) = args.d4 {
        let r = pbzarr_readers::D4Reader::open(d4)?;
        let g = r.contigs().clone();
        let total: u64 = g.contigs().iter().map(|c| c.length).sum();
        (g, total)
    } else {
        let contigs: Vec<Contig> = (0..args.contigs)
            .map(|i| Contig {
                name: format!("chr{}", i + 1),
                length: args.length,
            })
            .collect();
        let g = Reference::new(contigs)?;
        let total: u64 = g.contigs().iter().map(|c| c.length).sum();
        (g, total)
    };
    let _ = total_length;

    let tmp = TempDir::new()?;
    let path = args
        .out
        .clone()
        .unwrap_or_else(|| tmp.path().join("profile.pbz"));

    let mut dataset = Dataset::create(&path, genome.clone())?;

    let import_cfg = Config {
        workers: args.workers,
        chunk: args.chunk,
        shard: args.shard,
        shard_columns: args.shard_columns,
        ..Config::default()
    };

    let t0 = Instant::now();
    let report = if let Some(ref d4) = args.d4 {
        // `from_d4` creates the (int32) track itself.
        let sources: Vec<Source> = (0..args.cols)
            .map(|i| Source::labeled(d4.clone(), format!("s{i}")))
            .collect();
        from_d4(&mut dataset, "depth", &sources, import_cfg)?
    } else {
        let column_names: Vec<String> = (0..args.cols).map(|i| format!("s{i}")).collect();
        let mut cfg = TrackConfig::new(Dtype::U32)
            .columns(column_names)
            .column_dim("sample")
            .chunk(args.chunk)
            .shard_columns(args.shard_columns);
        cfg.shard = args.shard;
        let readers: Vec<SynthReader> = (0..args.cols)
            .map(|i| SynthReader::new(genome.clone(), 0x9E3779B9u32.wrapping_mul(i as u32 + 1)))
            .collect();
        Import::from_readers(readers)?
            .into_dataset(&mut dataset, vec![("depth".into(), cfg)])
            .readers_as_columns()
            .config(Config {
                workers: args.workers,
                ..Config::default()
            })
            .run()?
    };
    let elapsed = t0.elapsed();
    let secs = elapsed.as_secs_f64();

    let per_contig_total: u64 = genome.contigs().iter().map(|c| c.length).sum();
    let logical_bytes = (per_contig_total as usize) * args.cols * std::mem::size_of::<u32>();
    println!(
        "imported {} positions, {} chunks, {} logical bytes, wall {:.2}s",
        report.positions, report.chunks_written, logical_bytes, secs,
    );
    println!(
        "  throughput (logical): {}",
        human_rate(logical_bytes as u64, secs)
    );
    println!(
        "  throughput (reported): {}",
        human_rate(report.bytes_written, secs)
    );

    if args.keep || args.out.is_some() {
        println!("store kept at {}", path.display());
        std::mem::forget(tmp);
    }

    Ok(())
}
