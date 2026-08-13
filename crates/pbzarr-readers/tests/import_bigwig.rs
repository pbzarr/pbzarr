//! End-to-end: synthesize small bigWig files in-process (see `tests/common`),
//! import via `from_bigwig`, read back via `Track::read_region`.

mod common;

use std::path::Path;

use pbzarr::import::{Config, Source};
use pbzarr::{PbzStore, Region};
use pbzarr_readers::from_bigwig;
use tempfile::TempDir;

/// Banded intervals: positions [i*10, (i+1)*10) carry value `(i % 50) + 1 + base`.
fn banded(chrom: &str, len: u32, base: f32) -> Vec<(&str, u32, u32, f32)> {
    (0..(len / 10))
        .map(|i| (chrom, i * 10, (i + 1) * 10, ((i % 50) + 1) as f32 + base))
        .collect()
}

/// Count chunk data files under a contig's track array (everything but
/// `zarr.json`). zarrs does not write chunks that equal the fill value.
#[allow(dead_code)]
fn count_chunk_files(array_dir: &Path) -> usize {
    fn walk(dir: &Path, n: &mut usize) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, n);
            } else if path.file_name().and_then(|s| s.to_str()) != Some("zarr.json") {
                *n += 1;
            }
        }
    }
    let mut n = 0;
    walk(array_dir, &mut n);
    n
}

#[test]
fn import_one_bigwig_into_scalar_track() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "a",
        &[("chr1", 1000)],
        &banded("chr1", 1000, 0.0),
        false,
    );

    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_bigwig(&mut store, "signal", &[Source::new(bw)], Config::default()).unwrap();

    let track = store.track("signal").unwrap();
    assert_eq!(track.rank(), 1);
    let region = Region {
        contig: store.genome_for("signal").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 1_000,
    };
    let got = track.read_region::<f32>(&region).unwrap();
    let arr = got.into_dimensionality::<ndarray::Ix1>().unwrap();

    for i in 0..100u32 {
        let v = ((i % 50) + 1) as f32;
        for p in (i * 10)..((i + 1) * 10) {
            assert_eq!(arr[p as usize], v, "pos {p}: expected {v}");
        }
    }
}

/// A sharded track must import to the same bytes as an unsharded one. Length
/// (10_000) is not a multiple of the shard span (4_000), so the run exercises
/// full interior shards plus a partial edge shard, with >1 worker.
#[test]
fn sharded_scalar_import_matches_unsharded() {
    let dir = TempDir::new().unwrap();
    let bw = common::write_bigwig(
        dir.path(),
        "s",
        &[("chr1", 10_000)],
        &banded("chr1", 10_000, 0.0),
        false,
    );
    let src = || vec![Source::new(bw.clone())];
    let region = |store: &PbzStore| Region {
        contig: store.genome_for("signal").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 10_000,
    };

    let plain_path = dir.path().join("plain.pbz");
    let mut plain = PbzStore::create(&plain_path).unwrap();
    from_bigwig(
        &mut plain,
        "signal",
        &src(),
        Config {
            chunk_size: Some(1_000),
            ..Config::default()
        },
    )
    .unwrap();

    let sharded_path = dir.path().join("sharded.pbz");
    let mut sharded = PbzStore::create(&sharded_path).unwrap();
    from_bigwig(
        &mut sharded,
        "signal",
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
        .track("signal")
        .unwrap()
        .read_region::<f32>(&region(&plain))
        .unwrap();
    let sharded_data = sharded
        .track("signal")
        .unwrap()
        .read_region::<f32>(&region(&sharded))
        .unwrap();
    assert_eq!(plain_data, sharded_data);
}

/// Three distinct-valued bigWigs import into a 2D cohort track column-for-column.
#[test]
fn cohort_import_distinct_columns() {
    let dir = TempDir::new().unwrap();
    let sources: Vec<Source> = [0.0f32, 100.0, 200.0]
        .iter()
        .enumerate()
        .map(|(i, &base)| {
            Source::labeled(
                common::write_bigwig(
                    dir.path(),
                    &format!("s{i}"),
                    &[("chr1", 2_000)],
                    &banded("chr1", 2_000, base),
                    false,
                ),
                format!("s{i}"),
            )
        })
        .collect();

    let store_path = dir.path().join("cohort.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_bigwig(
        &mut store,
        "signal",
        &sources,
        Config {
            chunk_size: Some(1_000),
            // The column axis is generic; override the "sample" default.
            column_dim: Some("context".into()),
            ..Config::default()
        },
    )
    .unwrap();

    assert_eq!(store.track("signal").unwrap().column_dim(), Some("context"));
    let region = Region {
        contig: store.genome_for("signal").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 2_000,
    };
    let data = store
        .track("signal")
        .unwrap()
        .read_region::<f32>(&region)
        .unwrap();
    assert_eq!(data.shape(), &[2_000, 3]);
    // Column j carries the same banding offset by j*100.
    for j in 0..3 {
        // pos 0 is band 0 -> 1.0; pos 995 is band 99 -> (99 % 50) + 1 = 50.0.
        assert_eq!(data[[0, j]], 1.0 + (j as f32) * 100.0);
        assert_eq!(data[[995, j]], 50.0 + (j as f32) * 100.0);
    }
}

/// Positions with no bigWig coverage import as 0 ("no coverage" = 0 for
/// coverage/percent tracks): `from_bigwig` creates the track with a 0.0 fill.
#[test]
fn uncovered_positions_import_as_zero() {
    let dir = TempDir::new().unwrap();
    // Only [0, 100) covered; the rest of the 3_000 bp contig is gaps.
    let bw = common::write_bigwig(
        dir.path(),
        "sparse",
        &[("chr1", 3_000)],
        &[("chr1", 0, 100, 9.0)],
        false,
    );

    let store_path = dir.path().join("sparse.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_bigwig(
        &mut store,
        "signal",
        &[Source::new(bw)],
        Config {
            chunk_size: Some(1_000),
            ..Config::default()
        },
    )
    .unwrap();

    let track = store.track("signal").unwrap();
    let cid = store.genome_for("signal").unwrap().id("chr1").unwrap();

    // Covered [0, 100) keeps 9.0.
    let head = track
        .read_region::<f32>(&Region {
            contig: cid,
            start: 0,
            end: 100,
        })
        .unwrap();
    assert!(head.iter().all(|v| *v == 9.0), "covered head must be 9.0");

    // Everything past coverage reads back as 0, including a fully-uncovered tail
    // chunk.
    let tail = track
        .read_region::<f32>(&Region {
            contig: cid,
            start: 100,
            end: 3_000,
        })
        .unwrap();
    assert!(
        tail.iter().all(|v| *v == 0.0),
        "uncovered positions must be 0"
    );
}
