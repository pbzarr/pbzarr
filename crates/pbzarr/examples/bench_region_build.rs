//! Timing harness for `build_region_store`. Synthesises a 2D cohort source,
//! scatters non-overlapping peaks across it, and times the region build.
//!
//! Usage:
//!   cargo run --release --example bench_region_build -- \
//!     [total] [n_cols] [src_chunk] [n_peaks] [peak_width] [out_chunk] \
//!     [workers] [decode_workers] [write_workers]

use std::sync::Arc;
use std::time::Instant;

use ndarray::Array2;
use pbzarr::genome::{Contig, Genome};
use pbzarr::io::Dtype;
use pbzarr::region_store::{RegionBuildConfig, build_region_store};
use pbzarr::{PbzStore, Region, TrackConfig};

fn arg<T: std::str::FromStr>(i: usize, default: T) -> T {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn opt_arg<T: std::str::FromStr>(i: usize) -> Option<T> {
    std::env::args().nth(i).and_then(|s| s.parse().ok())
}

fn estimate_source_tasks(intervals: &[(String, u64, u64)], src_chunk: u64) -> usize {
    let src_chunk = src_chunk.max(1);
    let mut last: Option<(&str, u64)> = None;
    let mut count = 0usize;

    for (contig, start, end) in intervals {
        let mut cursor = *start;
        while cursor < *end {
            let chunk_start = (cursor / src_chunk) * src_chunk;
            if last != Some((contig.as_str(), chunk_start)) {
                count += 1;
                last = Some((contig.as_str(), chunk_start));
            }
            cursor = (chunk_start + src_chunk).min(*end);
        }
    }

    count
}

fn split_workers(
    workers: usize,
    decode_workers: Option<usize>,
    write_workers: Option<usize>,
    source_tasks: usize,
    write_tasks: usize,
) -> (usize, usize) {
    let total = workers.max(1);
    let source_cap = source_tasks.max(1);
    let write_cap = write_tasks.max(1);

    match (decode_workers, write_workers) {
        (Some(d), Some(w)) => (d.max(1).min(source_cap), w.max(1).min(write_cap)),
        (Some(d), None) => {
            let d = d.max(1).min(source_cap);
            let w = total.saturating_sub(d).max(1).min(write_cap);
            (d, w)
        }
        (None, Some(w)) => {
            let w = w.max(1).min(write_cap);
            let d = total.saturating_sub(w).max(1).min(source_cap);
            (d, w)
        }
        (None, None) => {
            let w = (total / 2).max(1).min(write_cap);
            let d = total.saturating_sub(w).max(1).min(source_cap);
            (d, w)
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total: u64 = arg(1, 4_000_000);
    let n_cols: usize = arg(2, 32);
    let src_chunk: usize = arg(3, 1_000_000);
    let n_peaks: usize = arg(4, 20_000);
    let peak_width: u64 = arg(5, 200);
    let out_chunk: usize = arg(6, 1_000_000);
    let workers: usize = arg(7, 4);
    let decode_workers: Option<usize> = opt_arg(8);
    let write_workers: Option<usize> = opt_arg(9);

    let dir = tempfile::tempdir()?;
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: total,
    }])?;

    // --- build the source cohort store (setup, not timed) ---
    let t_setup = Instant::now();
    let mut src = PbzStore::create(dir.path().join("src.pbz"))?;
    let cols: Vec<String> = (0..n_cols).map(|i| format!("s{i}")).collect();
    src.create_track(
        "cov",
        g.clone(),
        TrackConfig::new(Dtype::I32)
            .columns(cols)
            .column_dim("sample")
            .chunk_size(src_chunk),
    )?;
    {
        let t = src.track("cov").unwrap();
        let step = src_chunk as u64;
        let mut start = 0u64;
        while start < total {
            let end = (start + step).min(total);
            let len = (end - start) as usize;
            let mut a = Array2::<i32>::zeros((len, n_cols));
            // Realistic low-entropy coverage: a spatially autocorrelated shared
            // depth (random walk clamped 0..250) plus a small per-column offset.
            // Compresses like real fiber-seq depth, so decode cost is realistic
            // (a random fill is worst-case for zstd and misleads the bottleneck).
            let mut st: u64 = 0x9E3779B97F4A7C15 ^ start;
            let mut depth: i64 = 40;
            for r in 0..len {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                depth += ((st >> 60) as i64 & 0x7) - 3; // step in [-3,4]
                depth = depth.clamp(0, 250);
                for c in 0..n_cols {
                    a[[r, c]] = (depth + (c as i64 % 5)) as i32;
                }
            }
            t.write_region::<i32>(
                &Region {
                    contig: g.id("chr1").unwrap(),
                    start,
                    end,
                },
                a.into_dyn(),
            )?;
            start = end;
        }
    }
    let setup_s = t_setup.elapsed().as_secs_f64();

    // --- scatter non-overlapping peaks evenly across the genome ---
    let stride = (total / n_peaks as u64).max(peak_width + 1);
    let intervals: Vec<(String, u64, u64)> = (0..n_peaks)
        .map(|i| {
            let s = i as u64 * stride;
            ("chr1".to_owned(), s, s + peak_width)
        })
        .filter(|(_, _, e)| *e <= total)
        .collect();
    let in_peak: u64 = intervals.iter().map(|(_, s, e)| e - s).sum();

    let src = Arc::new(src);
    let source_tasks = estimate_source_tasks(&intervals, src_chunk as u64);
    let write_tasks = in_peak.div_ceil(out_chunk as u64) as usize;
    let (effective_decode_workers, effective_write_workers) = split_workers(
        workers,
        decode_workers,
        write_workers,
        source_tasks,
        write_tasks,
    );

    // --- baseline: plain contiguous parallel decode of the whole source ---
    // (import-style read: every source chunk decoded once, in order, no gather).
    let src_gb = total as f64 * n_cols as f64 * 4.0 / 1e9;
    let t_base = Instant::now();
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let bounds: Vec<(u64, u64)> = (0..total)
            .step_by(src_chunk)
            .map(|s| (s, (s + src_chunk as u64).min(total)))
            .collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..workers {
                let (src, bounds, next, g) = (&src, &bounds, &next, &g);
                sc.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= bounds.len() {
                            break;
                        }
                        let (s, e) = bounds[i];
                        let r = Region {
                            contig: g.id("chr1").unwrap(),
                            start: s,
                            end: e,
                        };
                        let a = src.track("cov").unwrap().read_region::<i32>(&r).unwrap();
                        std::hint::black_box(a.sum());
                    }
                });
            }
        });
    }
    let base_s = t_base.elapsed().as_secs_f64();

    // --- time the region build ---
    let mut out = PbzStore::create(dir.path().join("out.pbz"))?;
    let t_build = Instant::now();
    let report = build_region_store(
        src,
        &intervals,
        &mut out,
        RegionBuildConfig {
            chunk_size: Some(out_chunk),
            workers,
            decode_workers,
            write_workers,
            ..Default::default()
        },
    )?;
    let build_s = t_build.elapsed().as_secs_f64();

    let out_bytes = in_peak * n_cols as u64 * 4;
    // Evenly-spaced peaks touch every source chunk, so decode ≈ whole source.
    let src_chunks_touched = (total.div_ceil(src_chunk as u64)) as f64;
    let decode_bytes = src_chunks_touched * src_chunk as f64 * n_cols as f64 * 4.0;
    let in_peak_gb = in_peak as f64 * n_cols as f64 * 4.0 / 1e9;
    let amplification = decode_bytes / (in_peak_gb * 1e9);
    println!(
        "src: {total} pos x {n_cols} cols, src_chunk={src_chunk}, src={src_gb:.2} GB\n\
         peaks: {} x {peak_width}bp -> in_peak={in_peak} pos ({in_peak_gb:.2} GB), out_chunk={out_chunk}\n\
         tasks: {source_tasks} source-decode tasks, {write_tasks} output write units, workers={workers} -> decode={effective_decode_workers}, write={effective_write_workers}\n\
         setup wrote synthetic source in {setup_s:.2}s\n\
         BASELINE plain decode whole source: {src_gb:.2} GB / {base_s:.2}s = {:.1} GB/s\n\
         GATHER build: {build_s:.2}s, ~{:.2} GB decoded ({amplification:.1}x amplification) = {:.1} GB/s decode | write_units_done={}",
        intervals.len(),
        src_gb / base_s,
        decode_bytes / 1e9,
        decode_bytes / build_s / 1e9,
        report.tasks_completed,
    );
    let _ = out_bytes;
    Ok(())
}
