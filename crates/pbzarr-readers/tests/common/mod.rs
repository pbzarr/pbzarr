//! Shared fixture writers for the reader/import tests.
//!
//! Synthesizes tiny `.bw` files in-process with `bigtools::BigWigWrite`. That
//! writer needs a tokio runtime; a current-thread one with `channel_size = 0`
//! keeps it single-threaded, which is plenty for fixtures of a few intervals.
//! BED fixtures are BGZF-compressed and tabix-indexed in-process with noodles.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use bigtools::BigWigWrite;
use bigtools::beddata::BedParserStreamingIterator;
use noodles_bgzf as bgzf;
use noodles_core::Position;
use noodles_csi as csi;
use noodles_tabix as tabix;

/// Write a bigWig from `intervals` (`chrom, start, end, value`) over the given
/// `chrom_sizes`. Intervals must be sorted by start within each chrom; pass
/// `allow_out_of_order = true` when chroms themselves are not in sorted order.
#[allow(dead_code)]
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

/// Write a BGZF-compressed BED with a `#`-prefixed header and a tabix index.
/// `rows` are `(chrom, start, end, data_cells)`; the header names chrom/start/end
/// plus one entry per data cell. Returns the `.bed.gz` path (sibling `.tbi` next
/// to it). Rows MUST be coordinate-sorted.
#[allow(dead_code)]
pub fn write_bed_bgzip_tabix(
    dir: &Path,
    name: &str,
    header: &[&str],
    rows: &[(&str, u64, u64, Vec<&str>)],
) -> PathBuf {
    write_bed_impl(dir, name, Some(header), rows)
}

/// Headerless variant of `write_bed_bgzip_tabix`; pass empty data cells for BED3.
#[allow(dead_code)]
pub fn write_headerless_bed_bgzip_tabix(
    dir: &Path,
    name: &str,
    rows: &[(&str, u64, u64, Vec<&str>)],
) -> PathBuf {
    write_bed_impl(dir, name, None, rows)
}

fn write_bed_impl(
    dir: &Path,
    name: &str,
    header: Option<&[&str]>,
    rows: &[(&str, u64, u64, Vec<&str>)],
) -> PathBuf {
    let gz_path = dir.join(format!("{name}.bed.gz"));
    let mut writer = std::fs::File::create(&gz_path)
        .map(bgzf::io::Writer::new)
        .unwrap();

    let mut indexer = tabix::index::Indexer::default();
    indexer.set_header(csi::binning_index::index::header::Builder::bed().build());

    if let Some(header) = header {
        writeln!(writer, "#{}", header.join("\t")).unwrap();
    }

    let mut start_position = writer.virtual_position();
    for (chrom, start, end, cells) in rows {
        write!(writer, "{chrom}\t{start}\t{end}").unwrap();
        for c in cells {
            write!(writer, "\t{c}").unwrap();
        }
        writeln!(writer).unwrap();

        let end_position = writer.virtual_position();
        indexer
            .add_record(
                chrom,
                Position::try_from(usize::try_from(*start).unwrap() + 1).unwrap(),
                Position::try_from(usize::try_from(*end).unwrap()).unwrap(),
                csi::binning_index::index::reference_sequence::bin::Chunk::new(
                    start_position,
                    end_position,
                ),
            )
            .unwrap();
        start_position = end_position;
    }

    writer.finish().unwrap();

    let index = indexer.build();
    tabix::fs::write(dir.join(format!("{name}.bed.gz.tbi")), &index).unwrap();
    gz_path
}
