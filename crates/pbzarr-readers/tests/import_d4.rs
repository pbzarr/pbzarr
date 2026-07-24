//! End-to-end: synthesize a small d4 file via the system `d4tools` (skipped
//! if unavailable), import via `import::from_d4`, read back via `Track::read_region`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use pbzarr::import::Config;
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{D4Source, from_d4};
use tempfile::TempDir;

fn have_d4tools() -> bool {
    Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Write a tiny bedGraph-based d4 where positions [i*10, (i+1)*10) carry value
/// `(i % 50) + 1` (1-indexed bands of 10), then call `d4tools create --genome`.
fn write_synthetic_d4(tmp: &Path, chrom: &str, len: u32) -> std::path::PathBuf {
    let sizes_path = tmp.join("genome.sizes");
    let mut sf = std::fs::File::create(&sizes_path).unwrap();
    writeln!(sf, "{chrom}\t{len}").unwrap();
    drop(sf);

    let bg_path = tmp.join("data.bedgraph");
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for i in 0..(len / 10) {
        let s = i * 10;
        let e = (i + 1) * 10;
        let v = (i % 50) + 1;
        writeln!(bf, "{chrom}\t{s}\t{e}\t{v}").unwrap();
    }
    drop(bf);

    let d4_path = tmp.join("out.d4");
    let status = Command::new("d4tools")
        .args(["create", "--genome"])
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");
    d4_path
}

/// Like `write_synthetic_d4` but every value is offset by `base`, so distinct
/// 2D columns carry distinct data.
fn write_synthetic_d4_offset(tmp: &Path, tag: &str, chrom: &str, len: u32, base: i32) -> PathBuf {
    let sizes_path = tmp.join(format!("{tag}.sizes"));
    std::fs::write(&sizes_path, format!("{chrom}\t{len}\n")).unwrap();

    let bg_path = tmp.join(format!("{tag}.bedgraph"));
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for i in 0..(len / 10) {
        let s = i * 10;
        let e = (i + 1) * 10;
        let v = base + (i % 50) as i32 + 1;
        writeln!(bf, "{chrom}\t{s}\t{e}\t{v}").unwrap();
    }
    drop(bf);

    let d4_path = tmp.join(format!("{tag}.d4"));
    let status = Command::new("d4tools")
        .args(["create", "--genome"])
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");
    d4_path
}

/// A sharded track must import to the same bytes as an unsharded one. The
/// length (10_000) is not a multiple of the shard span (4_000), so the run
/// exercises full interior shards, multiple inner chunks per shard, and a
/// partial edge shard. Imported with >1 worker to confirm concurrent writes to
/// distinct shard files stay correct.
#[test]
fn sharded_scalar_import_matches_unsharded() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = write_synthetic_d4(dir.path(), "chr1", 10_000);
    let src = || {
        vec![D4Source {
            path: d4.clone(),
            sample_label: None,
        }]
    };
    let region = |store: &PbzStore| Region {
        contig: store.genome_for("depth").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 10_000,
    };

    let plain_path = dir.path().join("plain.pbz");
    let mut plain = PbzStore::create(&plain_path).unwrap();
    from_d4(
        &mut plain,
        "depth",
        &src(),
        Config {
            chunk_size: Some(1_000),
            ..Config::default()
        },
    )
    .unwrap();

    let sharded_path = dir.path().join("sharded.pbz");
    let mut sharded = PbzStore::create(&sharded_path).unwrap();
    from_d4(
        &mut sharded,
        "depth",
        &src(),
        Config {
            workers: 4,
            chunk_size: Some(1_000),
            shard_size: Some(4_000),
            ..Config::default()
        },
    )
    .unwrap();

    let plain_data = plain
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&plain))
        .unwrap();
    let sharded_data = sharded
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&sharded))
        .unwrap();
    assert_eq!(plain_data, sharded_data);
}

/// Same equivalence for a 2D track: three distinct-valued sources, a
/// sharded layout spanning multiple inner chunks plus a partial edge shard,
/// imported with >1 worker, must match the unsharded import column-for-column.
#[test]
fn sharded_two_d_import_matches_unsharded() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let sources: Vec<D4Source> = [0, 100, 200]
        .iter()
        .enumerate()
        .map(|(i, &base)| D4Source {
            path: write_synthetic_d4_offset(dir.path(), &format!("s{i}"), "chr1", 10_000, base),
            sample_label: Some(format!("s{i}")),
        })
        .collect();
    let region = |store: &PbzStore| Region {
        contig: store.genome_for("depth").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 10_000,
    };

    let plain_path = dir.path().join("plain.pbz");
    let mut plain = PbzStore::create(&plain_path).unwrap();
    from_d4(
        &mut plain,
        "depth",
        &sources,
        Config {
            chunk_size: Some(1_000),
            ..Config::default()
        },
    )
    .unwrap();

    let sharded_path = dir.path().join("sharded.pbz");
    let mut sharded = PbzStore::create(&sharded_path).unwrap();
    from_d4(
        &mut sharded,
        "depth",
        &sources,
        Config {
            workers: 4,
            chunk_size: Some(1_000),
            shard_size: Some(4_000),
            ..Config::default()
        },
    )
    .unwrap();

    let plain_data = plain
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&plain))
        .unwrap();
    let sharded_data = sharded
        .track("depth")
        .unwrap()
        .read_region::<i32>(&region(&sharded))
        .unwrap();
    assert_eq!(plain_data, sharded_data);
    // Sanity: the three columns really do differ (offsets applied).
    assert_ne!(sharded_data[[0, 0]], sharded_data[[0, 1]]);
    // A multi-source import defaults the column axis to "sample".
    assert_eq!(plain.track("depth").unwrap().column_dim(), Some("sample"));
}

#[test]
fn import_one_d4_into_scalar_track() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = write_synthetic_d4(dir.path(), "chr1", 1_000);

    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_d4(
        &mut store,
        "depth",
        &[D4Source {
            path: d4,
            sample_label: None,
        }],
        Config::default(),
    )
    .unwrap();

    let track = store.track("depth").unwrap();
    assert_eq!(track.rank(), 1);
    let region = Region {
        contig: store.genome_for("depth").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 1_000,
    };
    let got = track.read_region::<i32>(&region).unwrap();
    let arr = got.into_dimensionality::<ndarray::Ix1>().unwrap();

    for i in 0..100u32 {
        let v: i32 = ((i % 50) + 1) as i32;
        for p in (i * 10)..((i + 1) * 10) {
            assert_eq!(arr[p as usize], v, "pos {p}: expected {v}");
        }
    }
}

/// A multi-contig d4 imports each contig's data under the right name, even when
/// the file orders contigs unusually. The track genome is built from the d4
/// header, and the pipeline resolves each task's flat range back to a contig
/// name before handing it to the reader.
#[test]
fn import_d4_multi_contig_writes_correct_contigs() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();

    // d4 has chr2 first, chr1 second. chr2 values: 100. chr1 values: 7.
    let d4_path = dir.path().join("a.d4");
    let bg_path = d4_path.with_extension("bedgraph");
    let sizes_path = d4_path.with_extension("sizes");
    std::fs::write(&bg_path, "chr2\t0\t500\t100\nchr1\t0\t300\t7\n").unwrap();
    std::fs::write(&sizes_path, "chr2\t500\nchr1\t300\n").unwrap();
    let status = Command::new("d4tools")
        .args(["create", "--genome"])
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");

    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_d4(
        &mut store,
        "depth",
        &[D4Source {
            path: d4_path,
            sample_label: None,
        }],
        Config::default(),
    )
    .unwrap();

    let genome = store.genome_for("depth").unwrap();
    let chr1 = store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&Region {
            contig: genome.id("chr1").unwrap(),
            start: 0,
            end: 300,
        })
        .unwrap()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let chr2 = store
        .track("depth")
        .unwrap()
        .read_region::<i32>(&Region {
            contig: genome.id("chr2").unwrap(),
            start: 0,
            end: 500,
        })
        .unwrap()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    assert!(chr1.iter().all(|&v| v == 7), "chr1 should have value 7");
    assert!(chr2.iter().all(|&v| v == 100), "chr2 should have value 100");
}
