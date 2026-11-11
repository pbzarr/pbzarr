//! End-to-end: synthesize small bigWig files in-process (see `tests/common`),
//! import via `from_bigwig`, read back via `Dataset::read_region`.

mod common;

use std::path::Path;

use ndarray::ArrayD;
use pbzarr::Dataset;
use pbzarr::import::{Config, Source};
use pbzarr_readers::from_bigwig;
use tempfile::TempDir;

/// Banded intervals: positions [i*10, (i+1)*10) carry value `(i % 50) + 1 + base`.
fn banded(chrom: &str, len: u32, base: f32) -> Vec<(&str, u32, u32, f32)> {
    (0..(len / 10))
        .map(|i| (chrom, i * 10, (i + 1) * 10, ((i % 50) + 1) as f32 + base))
        .collect()
}

fn read(store: &Path) -> ArrayD<f32> {
    Dataset::open(store)
        .unwrap()
        .read_region::<f32>("signal", &"chr1".parse().unwrap())
        .unwrap()
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

    let store = dir.path().join("out.pbz");
    from_bigwig(
        store.as_path(),
        "signal",
        &[Source::new(bw)],
        None,
        Config::default(),
    )
    .unwrap();

    let arr = read(&store);
    assert_eq!(arr.shape(), [1_000]);
    for i in 0..100u32 {
        let v = ((i % 50) + 1) as f32;
        for p in (i * 10)..((i + 1) * 10) {
            assert_eq!(arr[p as usize], v, "pos {p}: expected {v}");
        }
    }
}

/// A sharded track must import to the same values as an unsharded one. Length
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

    let plain = dir.path().join("plain.pbz");
    from_bigwig(
        plain.as_path(),
        "signal",
        &src(),
        None,
        Config {
            chunk: 1_000,
            shard: None,
            ..Config::default()
        },
    )
    .unwrap();

    let sharded = dir.path().join("sharded.pbz");
    from_bigwig(
        sharded.as_path(),
        "signal",
        &src(),
        None,
        Config {
            workers: 4,
            chunk: 1_000,
            shard: Some(4),
            ..Config::default()
        },
    )
    .unwrap();

    let plain_data = read(&plain);
    assert_eq!(plain_data.shape(), [10_000]);
    assert_eq!(plain_data, read(&sharded));
}

fn import_gapped(
    dir: &Path,
    fill: Option<pbzarr::Fill>,
) -> (ArrayD<f32>, serde_json::Value, usize) {
    let bw = common::write_bigwig(
        dir,
        "gap",
        &[("chr1", 1000)],
        &[("chr1", 0, 10, 5.0)],
        false,
    );
    let store = dir.join("gap.pbz");
    from_bigwig(
        store.as_path(),
        "signal",
        &[Source::new(bw)],
        fill,
        Config {
            chunk: 100,
            shard: None,
            ..Config::default()
        },
    )
    .unwrap();
    let dataset = Dataset::open(&store).unwrap();
    let attrs = dataset.variables()["signal"].metadata["attributes"].clone();
    let stored = walkdir(&store.join("signal/c"));
    (read(&store), attrs, stored)
}

fn walkdir(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .map(|path| if path.is_dir() { walkdir(&path) } else { 1 })
        .sum()
}

#[test]
fn bigwig_gaps_default_to_the_null() {
    let dir = TempDir::new().unwrap();
    let (arr, attrs, stored) = import_gapped(dir.path(), None);
    assert!(arr.iter().take(10).all(|&v| v == 5.0));
    assert!(arr.iter().skip(10).all(|v| v.is_nan()));
    assert!(attrs["_FillValue"].is_string());
    assert_eq!(stored, 1, "chunks that hold only the fill are not stored");
}

#[test]
fn bigwig_fill_zero_writes_zero_at_gaps() {
    let dir = TempDir::new().unwrap();
    let fill = pbzarr::Fill::Value(serde_json::json!(0));
    let (arr, attrs, stored) = import_gapped(dir.path(), Some(fill));
    assert!(arr.iter().take(10).all(|&v| v == 5.0));
    assert!(arr.iter().skip(10).all(|&v| v == 0.0));
    assert!(attrs["_FillValue"].is_null());
    assert_eq!(stored, 1, "chunks that hold only the fill are not stored");
}
