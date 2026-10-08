//! Average busy threads of an import whose decode and encode are both
//! CPU-heavy, like BAM pileup into pcodec. Run under `/usr/bin/time` and
//! compare (user + sys) / real against `workers`.
//!
//! Usage: cargo run --release -p pbzarr --example import_threads -- <out.pbz> [workers]

use pbzarr::import::{Import, PipelineOptions};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ReaderError, ValueReader};
use pbzarr::{Contig, ExplicitArraySpec, Genome, PbzStore, TrackConfig};

const SPEC: &str = r#"{
  "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1048576, 8]}},
  "codecs": [{"name": "sharding_indexed", "configuration": {
    "chunk_shape": [16384, 8],
    "codecs": [
      {"name": "transpose", "configuration": {"order": [1, 0]}},
      {"name": "numcodecs.pcodec", "configuration": {"level": 8}}
    ],
    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
    "index_location": "end"}}]
}"#;

/// Depth-like values that cost a fixed amount of CPU per position to decode.
#[derive(Clone)]
struct BusyReader {
    genome: Genome,
    schema: OutputSchema,
    seed: u64,
}

impl ValueReader for BusyReader {
    fn contigs(&self) -> &Genome {
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
    ) -> Result<(), ReaderError> {
        let OutputSinkMut::U32(dst) = &mut outputs[0] else {
            panic!("u32 sink expected")
        };
        for pos in start..end {
            let mut x = pos ^ self.seed;
            for _ in 0..64 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
            }
            dst[(pos - start) as usize] = 25 + (x % 11) as u32;
        }
        Ok(())
    }

    fn fork(&self) -> Result<Self, ReaderError> {
        Ok(self.clone())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: import_threads <out.pbz> [workers]")?;
    let workers: usize = args.next().map_or(Ok(4), |s| s.parse())?;
    let samples = 16;
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 24_000_000,
    }])?;
    let mut store = PbzStore::create(&path)?;
    store.create_track(
        "depth",
        genome.clone(),
        TrackConfig::new(Dtype::U32)
            .columns((0..samples).map(|i| format!("s{i}")).collect())
            .column_dim("sample")
            .codecs(ExplicitArraySpec::parse(SPEC)?),
    )?;
    let readers = (0..samples as u64)
        .map(|seed| BusyReader {
            genome: genome.clone(),
            schema: OutputSchema::single("depth", Dtype::U32),
            seed,
        })
        .collect();
    Import::from_readers(readers)?
        .into_track(store.track("depth").ok_or("no depth track")?)
        .readers_as_columns()
        .options(PipelineOptions {
            workers,
            ..PipelineOptions::default()
        })
        .run()?;
    Ok(())
}
