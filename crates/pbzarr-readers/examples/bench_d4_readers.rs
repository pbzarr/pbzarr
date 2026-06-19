//!   cargo run --release --example bench_d4_readers -- <path.d4> [--chunk-size N] [--iters K]

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use d4::ptab::{BitArrayPartReader, DecodeResult, Decoder};
use d4::stab::{RangeRecord, SecondaryTablePartReader, SparseArrayPartReader};

fn parse_args() -> (PathBuf, usize, usize) {
    let mut it = std::env::args().skip(1);
    let path = PathBuf::from(
        it.next()
            .expect("usage: bench_d4_readers <path.d4> [--chunk-size N] [--iters K]"),
    );
    let mut chunk_size: usize = 1_000_000;
    let mut iters: usize = 3;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--chunk-size" => chunk_size = it.next().unwrap().parse().unwrap(),
            "--iters" => iters = it.next().unwrap().parse().unwrap(),
            other => panic!("unknown arg: {other}"),
        }
    }
    (path, chunk_size, iters)
}

struct Region {
    chrom: String,
    len: u32,
}

fn list_regions_mmap(path: &PathBuf) -> Vec<Region> {
    let r: d4::D4TrackReader = d4::D4TrackReader::open_first_track(path).expect("open mmap reader");
    r.header()
        .chrom_list()
        .iter()
        .map(|c| Region {
            chrom: c.name.clone(),
            len: c.size as u32,
        })
        .collect()
}

fn bench_mmap(path: &PathBuf, chunk: usize, regions: &[Region]) -> (f64, u64, u64) {
    // Open once, scan whole genome in chunk-sized requests using the same
    // split + decode_block pattern as the pbzarr D4Reader.
    let mut r: d4::D4TrackReader =
        d4::D4TrackReader::open_first_track(path).expect("open mmap reader");
    let mut total_positions: u64 = 0;
    let mut checksum: u64 = 0;
    let t0 = Instant::now();
    for reg in regions {
        let mut start: u32 = 0;
        while start < reg.len {
            let end = (start + chunk as u32).min(reg.len);
            let span = (end - start) as usize;
            let mut buf: Vec<u32> = vec![0; span];

            let parts: Vec<(BitArrayPartReader, SparseArrayPartReader<RangeRecord>)> =
                r.split(Some(span.max(1))).expect("split");
            for (mut ptab, mut stab) in parts {
                let (part_chr, part_begin, part_end) = {
                    let (c, b, e) = ptab.region();
                    (c.to_string(), b, e)
                };
                if part_chr != reg.chrom {
                    continue;
                }
                let from = start.max(part_begin);
                let to = end.min(part_end);
                if from >= to {
                    continue;
                }
                let mut codec = ptab.to_codec();
                codec.decode_block(from as usize, (to - from) as usize, |pos, value| {
                    let resolved = match value {
                        DecodeResult::Definitely(v) => v,
                        DecodeResult::Maybe(v) => stab.decode(pos as u32).unwrap_or(v),
                    };
                    buf[pos - start as usize] = resolved as u32;
                });
            }
            for v in &buf {
                checksum = checksum.wrapping_add(*v as u64);
            }
            total_positions += span as u64;
            start = end;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    (secs, total_positions, checksum)
}

fn bench_ssio(path: &PathBuf, chunk: usize, regions: &[Region]) -> (f64, u64, u64) {
    use d4::ssio::D4TrackReader as SsioReader;
    let file = File::open(path).expect("open file");
    let mut r = SsioReader::from_reader(file, None).expect("open ssio reader");
    let mut total_positions: u64 = 0;
    let mut checksum: u64 = 0;
    let t0 = Instant::now();
    for reg in regions {
        let mut start: u32 = 0;
        while start < reg.len {
            let end = (start + chunk as u32).min(reg.len);
            let span = (end - start) as usize;
            let mut buf: Vec<u32> = vec![0; span];
            let mut view = r.get_view(&reg.chrom, start, end).expect("get_view");
            for (i, slot) in buf.iter_mut().enumerate() {
                let (pos, value) = view.next().expect("expected value").expect("read value");
                debug_assert_eq!(pos, start + i as u32);
                *slot = value as u32;
            }
            for v in &buf {
                checksum = checksum.wrapping_add(*v as u64);
            }
            total_positions += span as u64;
            start = end;
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    (secs, total_positions, checksum)
}

fn rate(bytes: u64, secs: f64) -> f64 {
    (bytes as f64) / 1_048_576.0 / secs
}

fn main() {
    let (path, chunk, iters) = parse_args();
    let regions = list_regions_mmap(&path);
    let total_positions_per_iter: u64 = regions.iter().map(|r| r.len as u64).sum();
    let bytes_per_iter = total_positions_per_iter * std::mem::size_of::<u32>() as u64;
    println!(
        "file: {} | {} contigs, {} total positions, chunk={}, iters={}",
        path.display(),
        regions.len(),
        total_positions_per_iter,
        chunk,
        iters,
    );

    // Warm the page cache once for fairness.
    let (_, _, _) = bench_mmap(&path, chunk, &regions);

    let mut mmap_times = Vec::with_capacity(iters);
    let mut ssio_times = Vec::with_capacity(iters);
    let mut last_mmap_checksum: u64 = 0;
    let mut last_ssio_checksum: u64 = 0;

    for i in 0..iters {
        let (s, _, c) = bench_mmap(&path, chunk, &regions);
        mmap_times.push(s);
        last_mmap_checksum = c;
        let (s, _, c) = bench_ssio(&path, chunk, &regions);
        ssio_times.push(s);
        last_ssio_checksum = c;
        println!(
            "  iter {}: mmap={:.3}s ({:.1} MiB/s)   ssio={:.3}s ({:.1} MiB/s)",
            i + 1,
            mmap_times[i],
            rate(bytes_per_iter, mmap_times[i]),
            ssio_times[i],
            rate(bytes_per_iter, ssio_times[i]),
        );
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mmap_avg = mean(&mmap_times);
    let ssio_avg = mean(&ssio_times);
    println!("---");
    println!(
        "mmap avg: {:.3}s ({:.1} MiB/s)",
        mmap_avg,
        rate(bytes_per_iter, mmap_avg),
    );
    println!(
        "ssio avg: {:.3}s ({:.1} MiB/s)",
        ssio_avg,
        rate(bytes_per_iter, ssio_avg),
    );
    println!("speedup mmap/ssio: {:.2}x", ssio_avg / mmap_avg);
    assert_eq!(
        last_mmap_checksum, last_ssio_checksum,
        "checksums differ — readers disagree"
    );
    println!("checksums agree ({})", last_mmap_checksum);
}
