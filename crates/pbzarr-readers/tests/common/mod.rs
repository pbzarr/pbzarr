//! Shared bigWig fixture writer for the reader/import tests.
//!
//! Synthesizes tiny `.bw` files in-process with `bigtools::BigWigWrite`. That
//! writer needs a tokio runtime; a current-thread one with `channel_size = 0`
//! keeps it single-threaded, which is plenty for fixtures of a few intervals.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use bigtools::BigWigWrite;
use bigtools::beddata::BedParserStreamingIterator;

/// Write a bigWig from `intervals` (`chrom, start, end, value`) over the given
/// `chrom_sizes`. Intervals must be sorted by start within each chrom; pass
/// `allow_out_of_order = true` when chroms themselves are not in sorted order.
pub fn write_bigwig(
    tmp: &Path,
    name: &str,
    chrom_sizes: &[(&str, u32)],
    intervals: &[(&str, u32, u32, f32)],
    allow_out_of_order: bool,
) -> PathBuf {
    let bg_path = tmp.join(format!("{name}.bedgraph"));
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for (c, s, e, v) in intervals {
        writeln!(bf, "{c}\t{s}\t{e}\t{v}").unwrap();
    }
    drop(bf);

    let chrom_map: HashMap<String, u32> = chrom_sizes
        .iter()
        .map(|(c, l)| ((*c).to_string(), *l))
        .collect();

    let bw_path = tmp.join(format!("{name}.bw"));
    let infile = std::fs::File::open(&bg_path).unwrap();
    let vals = BedParserStreamingIterator::from_bedgraph_file(infile, allow_out_of_order);

    let mut out = BigWigWrite::create_file(&bw_path, chrom_map).unwrap();
    out.options.channel_size = 0;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    out.write(vals, runtime).unwrap();
    bw_path
}
