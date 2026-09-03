//! End-to-end: import the committed d4 fixtures under `fixtures/d4/` via
//! `import::from_d4`, read back via `Track::read_region`.

use std::path::{Path, PathBuf};

use pbzarr::import::{Config, Source};
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{D4Reader, from_d4};
use tempfile::TempDir;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name)
}

/// Sharded/unsharded equivalence for a 2D cohort track: three distinct-valued
/// sources, a sharded layout spanning multiple inner chunks plus a partial
/// edge shard, imported with >1 worker, must match the unsharded import
/// column-for-column.
#[test]
fn sharded_cohort_import_matches_unsharded() {
    let dir = TempDir::new().unwrap();
    let sources: Vec<Source> = (0..3)
        .map(|i| Source::labeled(fixture(&format!("cohort_s{i}.d4")), format!("s{i}")))
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
    let dir = TempDir::new().unwrap();
    let d4 = fixture("banded_1k.d4");

    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_d4(&mut store, "depth", &[Source::new(d4)], Config::default()).unwrap();

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
    let dir = TempDir::new().unwrap();
    // Fixture has chr2 first, chr1 second. chr2 values: 100. chr1 values: 7.
    let d4_path = fixture("multi_contig.d4");

    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
    from_d4(
        &mut store,
        "depth",
        &[Source::new(d4_path)],
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

/// Fix-point (non-integral) d4 files are rejected at open time.
#[test]
fn fixed_point_d4_is_rejected() {
    let err = D4Reader::open(fixture("fixed_point.d4"))
        .err()
        .expect("expected error when opening fixed-point d4");
    let msg = err.to_string();
    assert!(
        msg.contains("fix-point"),
        "error message should contain 'fix-point', got: {msg}"
    );
}
