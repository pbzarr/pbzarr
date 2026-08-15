use std::sync::Arc;

use ndarray::{Array1, ArrayD};
use pbzarr::genome::Contig;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzError, PbzStore, TrackConfig};
use tempfile::TempDir;
use zarrs::storage::ReadableWritableListableStorage;
use zarrs::storage::store::MemoryStore;

fn tiny_genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 1000,
        },
        Contig {
            name: "chr2".into(),
            length: 500,
        },
    ])
    .unwrap()
    .with_name("test")
}

#[test]
fn create_writes_bare_marker_and_reopens_empty() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let store = PbzStore::create(&path).unwrap();
    assert_eq!(store.track_names().count(), 0);

    let reopened = PbzStore::open(&path).unwrap();
    assert_eq!(reopened.track_names().count(), 0);
}

#[test]
fn create_track_builds_flat_group() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", tiny_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let store = PbzStore::open(&path).unwrap();
    let t = store.track("depth").expect("track discovered");
    assert_eq!(t.dtype(), Dtype::I32);
    assert_eq!(t.rank(), 1);
    // flat values length == ΣL == 1500
    assert_eq!(t.total_len(), 1500);
    assert_eq!(t.genome().checksum(), tiny_genome().checksum());
}

#[test]
fn create_track_rejects_invalid_direct_child_names_before_writing() {
    for invalid in ["", "__axis", "...", "nested/track"] {
        let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let mut store = PbzStore::create_with_storage(storage).unwrap();

        let result = store.create_track(invalid, tiny_genome(), TrackConfig::new(Dtype::I32));
        let error = match result {
            Ok(_) => panic!("track names must be valid non-root Zarr node names"),
            Err(error) => error.to_string(),
        };

        assert!(error.contains("invalid PBZ/Zarr node name"), "{error}");
        assert!(error.contains(&format!("{invalid:?}")), "{error}");
        assert!(store.track_names().next().is_none());
    }
}

#[test]
fn open_discovers_only_conforming_groups() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", tiny_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let reopened = PbzStore::open(&path).unwrap();
    let names: Vec<&str> = reopened.track_names().collect();
    assert_eq!(names, vec!["depth"]);
    let g = reopened.track("depth").unwrap().genome();
    assert_eq!(g.name(), Some("test"));
    assert_eq!(g.contigs().len(), 2);
    assert_eq!(g.offsets(), vec![0, 1000, 1500]);
}

/// Proves `PbzStore`/`Track` are storage-agnostic: a `MemoryStore` (no
/// filesystem involved) round-trips through the same public API as
/// `create`/`open`. This is the seam a future async-to-sync remote adapter
/// slots into unchanged.
#[test]
fn roundtrips_over_memory_store() {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());

    let mut store = PbzStore::create_with_storage(storage.clone()).unwrap();
    store
        .create_track("depth", tiny_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr2:100-110".parse().unwrap())
        .unwrap();
    let data: ArrayD<i32> = Array1::from(vec![42i32; 10]).into_dyn();
    store
        .track("depth")
        .unwrap()
        .write_region(&region, data)
        .unwrap();

    // Reopen from the same MemoryStore handle to prove discovery works over
    // a non-filesystem backend too (the store is dropped and rebuilt, but
    // the backing MemoryStore Arc keeps the data alive).
    drop(store);
    let reopened = PbzStore::open_with_storage(storage).unwrap();
    let track = reopened.track("depth").unwrap();
    let region = reopened
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr2:100-110".parse().unwrap())
        .unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vec![42i32; 10]);
}

#[test]
fn failed_population_leaves_tracks_unpublished_and_blocks_same_name_retry() {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let mut store = PbzStore::create_with_storage(storage.clone()).unwrap();

    let result = store.create_tracks_with(
        vec![
            (
                "depth".to_owned(),
                tiny_genome(),
                TrackConfig::new(Dtype::I32),
            ),
            (
                "mask".to_owned(),
                tiny_genome(),
                TrackConfig::new(Dtype::U8),
            ),
        ],
        |_tracks| Err::<(), _>(PbzError::Store("population failed".into())),
    );

    assert!(result.is_err());
    assert!(store.track("depth").is_none());
    assert!(store.track("mask").is_none());
    let reopened = PbzStore::open_with_storage(storage).unwrap();
    assert!(reopened.track("depth").is_none());
    assert!(reopened.track("mask").is_none());

    let retry = store.create_track("depth", tiny_genome(), TrackConfig::new(Dtype::I32));
    assert!(retry.is_err(), "incomplete physical group must block retry");
}

#[test]
fn successful_population_publishes_only_after_data_is_written() {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let mut store = PbzStore::create_with_storage(storage.clone()).unwrap();
    let population_storage = storage.clone();

    store
        .create_tracks_with(
            vec![(
                "depth".to_owned(),
                tiny_genome(),
                TrackConfig::new(Dtype::I32),
            )],
            |tracks| {
                let during_population = PbzStore::open_with_storage(population_storage.clone())?;
                assert!(during_population.track("depth").is_none());

                let region = tracks[0].genome().resolve(&"chr1:0-4".parse().unwrap())?;
                tracks[0].write_region(&region, Array1::from(vec![7i32; 4]).into_dyn())
            },
        )
        .unwrap();

    assert!(store.track("depth").is_some());
    let reopened = PbzStore::open_with_storage(storage).unwrap();
    let track = reopened.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-4".parse().unwrap())
        .unwrap();
    assert_eq!(
        track
            .read_region::<i32>(&region)
            .unwrap()
            .into_raw_vec_and_offset()
            .0,
        vec![7; 4]
    );
}
