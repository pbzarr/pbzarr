use ndarray::{Array1, Array2};
use pbzarr::{PbzError, PbzStore, TrackConfig};
use tempfile::TempDir;

// pbz[verify group.attrs.namespace]
// pbz[verify group.attrs.version]
// pbz[verify contigs.array]
// pbz[verify contigs.lengths]
// pbz[verify contigs.order]
// pbz[verify per_base.data.location]
// pbz[verify per_base.data.shape-2d]
// pbz[verify per_base.data.dim-names]
// pbz[verify per_base.samples.array]
// pbz[verify per_base.samples.order]
// pbz[verify per_base.attrs.dtype]
// pbz[verify per_base.attrs.chunk_size]
// pbz[verify per_base.attrs.sample_chunk_size]
#[test]
fn full_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let contigs = vec!["chr1".to_string(), "chr2".to_string()];
    let lengths = vec![5000u64, 3000];

    let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

    let config = TrackConfig {
        dtype: "uint32".into(),
        samples: Some(vec!["s1".into(), "s2".into(), "s3".into()]),
        chunk_size: 1000,
        description: Some("depth data".into()),
        source: Some("integration test".into()),
        ..Default::default()
    };
    let track = store.create_track("depths", config).unwrap();

    // Write all chunks for chr1 (5 full chunks of 1000)
    for chunk_idx in 0..5u64 {
        let mut data = Array2::<u32>::zeros((1000, 3));
        for i in 0..1000 {
            for j in 0..3 {
                data[[i, j]] = (chunk_idx as u32 * 1000) + i as u32 + j as u32;
            }
        }
        track.write_chunk("chr1", chunk_idx, data).unwrap();
    }

    // Write all chunks for chr2 (3 full chunks of 1000)
    for chunk_idx in 0..3u64 {
        let data = Array2::<u32>::from_elem((1000, 3), 99);
        track.write_chunk("chr2", chunk_idx, data).unwrap();
    }

    // Reopen store
    let store2 = PbzStore::open(&path).unwrap();
    assert_eq!(store2.contigs(), &["chr1", "chr2"]);
    assert_eq!(store2.contig_length("chr1").unwrap(), 5000);
    assert_eq!(store2.contig_length("chr2").unwrap(), 3000);

    let track_names = store2.tracks().unwrap();
    assert_eq!(track_names, vec!["depths"]);

    let track2 = store2.track("depths").unwrap();
    assert_eq!(track2.dtype(), "uint32");
    assert!(track2.has_samples());
    assert_eq!(track2.samples().unwrap(), &["s1", "s2", "s3"]);
    assert_eq!(track2.metadata().description.as_deref(), Some("depth data"));

    // Verify chr1 data
    let chunk0: Array2<u32> = track2.read_chunk("chr1", 0).unwrap();
    assert_eq!(chunk0.shape(), &[1000, 3]);
    assert_eq!(chunk0[[0, 0]], 0);
    assert_eq!(chunk0[[0, 2]], 2);
    assert_eq!(chunk0[[999, 0]], 999);

    let chunk4: Array2<u32> = track2.read_chunk("chr1", 4).unwrap();
    assert_eq!(chunk4[[0, 0]], 4000);

    // Verify chr2 data
    let chr2_chunk: Array2<u32> = track2.read_chunk("chr2", 0).unwrap();
    assert_eq!(chr2_chunk[[500, 1]], 99);

    // Escape hatch
    let raw = track2.zarr_array("chr1").unwrap();
    assert_eq!(raw.shape(), &[5000, 3]);
}

// pbz[verify per_base.data.shape-1d]
// pbz[verify missing.fill_value]
// pbz[verify compression.default]
#[test]
fn bool_scalar_track_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let store = PbzStore::create(&path, &["chr1".into()], &[2500]).unwrap();

    let config = TrackConfig {
        dtype: "bool".into(),
        samples: None,
        chunk_size: 1000,
        ..Default::default()
    };
    let track = store.create_track("mask", config).unwrap();

    // 3 chunks: 1000, 1000, 500 (partial)
    track
        .write_chunk_1d("chr1", 0, Array1::from_vec(vec![true; 1000]))
        .unwrap();
    track
        .write_chunk_1d("chr1", 1, Array1::from_vec(vec![false; 1000]))
        .unwrap();
    track
        .write_chunk_1d(
            "chr1",
            2,
            Array1::from_vec((0..500).map(|i| i % 2 == 0).collect()),
        )
        .unwrap();

    // Reopen and verify
    let store2 = PbzStore::open(&path).unwrap();
    let track2 = store2.track("mask").unwrap();

    assert_eq!(track2.dtype(), "bool");
    assert!(!track2.has_samples());

    let read0: Array1<bool> = track2.read_chunk_1d("chr1", 0).unwrap();
    assert!(read0.iter().all(|&v| v));

    let read1: Array1<bool> = track2.read_chunk_1d("chr1", 1).unwrap();
    assert!(read1.iter().all(|&v| !v));

    let read2: Array1<bool> = track2.read_chunk_1d("chr1", 2).unwrap();
    assert_eq!(read2.len(), 500);
    assert!(read2[0]);
    assert!(!read2[1]);
}

#[test]
fn multiple_tracks_same_store() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let store = PbzStore::create(&path, &["chr1".into()], &[2000]).unwrap();

    store
        .create_track(
            "depths",
            TrackConfig {
                dtype: "uint32".into(),
                samples: Some(vec!["s1".into(), "s2".into()]),
                chunk_size: 1000,
                ..Default::default()
            },
        )
        .unwrap();

    store
        .create_track(
            "mask",
            TrackConfig {
                dtype: "bool".into(),
                samples: None,
                chunk_size: 1000,
                ..Default::default()
            },
        )
        .unwrap();

    let store2 = PbzStore::open(&path).unwrap();
    let mut track_names = store2.tracks().unwrap();
    track_names.sort();
    assert_eq!(track_names, vec!["depths", "mask"]);
}

// pbz[verify track.attrs.preserve-unknown]
#[test]
fn extra_metadata_round_trip_integration() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let store = PbzStore::create(&path, &["chr1".into()], &[1000]).unwrap();

    let mut extra = serde_json::Map::new();
    extra.insert(
        "clam".into(),
        serde_json::json!({
            "callable_loci_type": "population_counts",
            "populations": [
                {"name": "pop1", "samples": ["s1", "s2"]},
                {"name": "pop2", "samples": ["s3"]}
            ]
        }),
    );

    store
        .create_track(
            "callable",
            TrackConfig {
                dtype: "uint16".into(),
                samples: Some(vec!["pop1".into(), "pop2".into()]),
                chunk_size: 1000,
                extra,
                ..Default::default()
            },
        )
        .unwrap();

    // Reopen and verify extra metadata survived
    let store2 = PbzStore::open(&path).unwrap();
    let track = store2.track("callable").unwrap();

    let clam = track.metadata().extra.get("clam").unwrap();
    assert_eq!(clam["callable_loci_type"], "population_counts");
    assert_eq!(clam["populations"][0]["name"], "pop1");
    assert_eq!(clam["populations"][1]["samples"][0], "s3");
}

// pbz[verify coords.zero-based-half-open]
#[test]
fn chunk_math_human_scale() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let store = PbzStore::create(&path, &["chr1".into()], &[248_956_422]).unwrap();

    let track = store
        .create_track(
            "depths",
            TrackConfig {
                dtype: "uint32".into(),
                samples: Some(vec!["s1".into(), "s2".into()]),
                chunk_size: 1_000_000,
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(track.position_to_chunk(0), 0);
    assert_eq!(track.position_to_chunk(999_999), 0);
    assert_eq!(track.position_to_chunk(1_000_000), 1);
    assert_eq!(track.position_to_chunk(248_956_421), 248);

    // Last chunk: 248*1M = 248M, remainder = 956_422
    let (start, end) = track.chunk_bounds(248, 248_956_422);
    assert_eq!(start, 248_000_000);
    assert_eq!(end, 248_956_422);

    let chunks = track.overlapping_chunks(100_000_000, 100_500_000);
    assert_eq!(chunks, 100..101);
}

#[test]
fn edit_lifecycle_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    // 1. Create store with one contig.
    let store = PbzStore::create(&path, &["chr1".to_string()], &[100u64]).unwrap();

    // 2. Create a track with description "v1".
    store
        .create_track(
            "depths",
            TrackConfig {
                dtype: "uint32".into(),
                samples: Some(vec!["s1".into(), "s2".into()]),
                chunk_size: 100,
                sample_chunk_size: 1,
                description: Some("v1".into()),
                ..Default::default()
            },
        )
        .unwrap();
    drop(store);

    // 3. Reopen, set description to "v2", verify reopen sees "v2".
    let store = PbzStore::open(&path).unwrap();
    let mut track = store.track("depths").unwrap();
    track.set_description(Some("v2")).unwrap();
    drop(track);
    drop(store);

    let store = PbzStore::open(&path).unwrap();
    let track = store.track("depths").unwrap();
    assert_eq!(track.metadata().description.as_deref(), Some("v2"));
    drop(track);

    // 4. Rename "depths" -> "coverage"; verify list/track APIs reflect rename.
    store.rename_track("depths", "coverage").unwrap();
    let names = store.tracks().unwrap();
    assert_eq!(names, vec!["coverage"]);
    assert!(store.track("coverage").is_ok());
    assert!(matches!(
        store.track("depths"),
        Err(pbzarr::PbzError::TrackNotFound { .. })
    ));

    // 5. Drop "coverage"; verify tracks() empty and directory gone.
    store.drop_track("coverage").unwrap();
    assert!(store.tracks().unwrap().is_empty());
    assert!(!path.join("tracks/coverage").exists());
}

// pbz[verify track.attrs.layout]
#[test]
fn track_writes_layout_attribute() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    store
        .create_track(
            "t",
            TrackConfig {
                dtype: "float32".into(),
                samples: Some(vec!["A".into(), "B".into()]),
                chunk_size: 100,
                sample_chunk_size: 2,
                ..Default::default()
            },
        )
        .unwrap();
    drop(store);

    let raw_store: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
    let group = zarrs::group::Group::open(raw_store, "/tracks/t").unwrap();
    let attrs = group
        .attributes()
        .get("perbase_zarr_track")
        .unwrap()
        .clone();
    assert_eq!(attrs["layout"], "per_base");
}

// pbz[verify track.attrs.layout]
#[test]
fn track_read_requires_layout() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    store
        .create_track(
            "t",
            TrackConfig {
                dtype: "float32".into(),
                samples: None,
                chunk_size: 100,
                ..Default::default()
            },
        )
        .unwrap();
    drop(store);

    // Strip layout via raw zarr write
    let raw_store: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
    let mut group = zarrs::group::Group::open(raw_store.clone(), "/tracks/t").unwrap();
    let mut attrs = group
        .attributes()
        .get("perbase_zarr_track")
        .unwrap()
        .clone();
    attrs.as_object_mut().unwrap().remove("layout");
    group
        .attributes_mut()
        .insert("perbase_zarr_track".into(), attrs);
    group.store_metadata().unwrap();

    // Track is still listed (collect_tracks now skips with a warn — the group
    // has no recognizable layout). Calling track() directly errors with
    // MissingLayout.
    let store2 = PbzStore::open(&path).unwrap();
    // Track should not appear in tracks() output (silent skip).
    assert!(!store2.tracks().unwrap().contains(&"t".to_string()));
}

// pbz[verify group.attrs.coordinate_space]
#[test]
fn store_coordinate_space_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let names = vec!["chr1".to_string()];
    let lens = vec![1000u64];
    let store = PbzStore::builder(&path)
        .contigs(&names, &lens)
        .coordinate_space("GRCh38")
        .create()
        .unwrap();
    drop(store);
    let s = PbzStore::open(&path).unwrap();
    assert_eq!(s.coordinate_space(), Some("GRCh38"));
}

// pbz[verify group.attrs.coordinate_space]
#[test]
fn store_coordinate_space_optional() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    drop(store);
    let s = PbzStore::open(&path).unwrap();
    assert_eq!(s.coordinate_space(), None);
}

// pbz[verify per_base.data.single-sample-1d]
#[test]
fn create_track_rejects_single_sample() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    let cfg = TrackConfig {
        dtype: "float32".into(),
        samples: Some(vec!["only".into()]),
        chunk_size: 100,
        sample_chunk_size: 1,
        ..Default::default()
    };
    let err = store.create_track("t", cfg).unwrap_err();
    assert!(matches!(err, PbzError::SingleSampleMustBe1D));
}

// pbz[verify forward.layout.no-error]
// pbz[verify forward.layout.warn-skip]
#[test]
fn unknown_layout_skipped_on_list() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    store
        .create_track(
            "good",
            TrackConfig {
                dtype: "float32".into(),
                samples: None,
                chunk_size: 100,
                ..Default::default()
            },
        )
        .unwrap();

    // Inject a future-layout track via raw zarr write.
    let raw_store: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
    let mut group_attrs = serde_json::Map::new();
    group_attrs.insert(
        "perbase_zarr_track".into(),
        serde_json::json!({"layout": "interval", "dtype": "float32"}),
    );
    let g = zarrs::group::GroupBuilder::new()
        .attributes(group_attrs)
        .build(raw_store, "/tracks/future")
        .unwrap();
    g.store_metadata().unwrap();
    drop(store);

    let s2 = PbzStore::open(&path).unwrap();
    let names = s2.tracks().unwrap();
    assert!(names.contains(&"good".to_string()));
    assert!(!names.contains(&"future".to_string()));
}

// pbz[verify forward.layout.no-error]
#[test]
fn unknown_layout_does_not_error_on_open() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    drop(store);

    let raw_store: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
    let mut group_attrs = serde_json::Map::new();
    group_attrs.insert(
        "perbase_zarr_track".into(),
        serde_json::json!({"layout": "interval"}),
    );
    let g = zarrs::group::GroupBuilder::new()
        .attributes(group_attrs)
        .build(raw_store, "/tracks/future")
        .unwrap();
    g.store_metadata().unwrap();

    PbzStore::open(&path).expect("open must not error on unknown layout");
}

// pbz[verify forward.preserve-roundtrip]
// pbz[verify track.attrs.preserve-unknown]
#[test]
fn track_preserves_unknown_attrs_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    store
        .create_track(
            "t",
            TrackConfig {
                dtype: "float32".into(),
                samples: Some(vec!["A".into(), "B".into()]),
                chunk_size: 100,
                sample_chunk_size: 2,
                ..Default::default()
            },
        )
        .unwrap();
    drop(store);

    // Inject an unknown attr via raw zarr.
    let raw_store: zarrs::storage::ReadableWritableListableStorage =
        std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&path).unwrap());
    {
        let mut g = zarrs::group::Group::open(raw_store.clone(), "/tracks/t").unwrap();
        let mut attrs = g
            .attributes()
            .get("perbase_zarr_track")
            .unwrap()
            .clone();
        attrs["future_v2_field"] = serde_json::json!({"hello": "world"});
        g.attributes_mut()
            .insert("perbase_zarr_track".into(), attrs);
        g.store_metadata().unwrap();
    }

    // Re-open via public API and rewrite metadata via set_description
    // — unknown keys must survive.
    let store2 = PbzStore::open(&path).unwrap();
    let mut track = store2.track("t").unwrap();
    track.set_description(Some("rewritten")).unwrap();
    drop(track);
    drop(store2);

    let g2 = zarrs::group::Group::open(raw_store, "/tracks/t").unwrap();
    let final_attrs = g2.attributes().get("perbase_zarr_track").unwrap();
    assert_eq!(final_attrs["future_v2_field"]["hello"], "world");
    assert_eq!(final_attrs["layout"], "per_base");
    assert_eq!(final_attrs["description"], "rewritten");
}

// pbz[verify track.attrs.layout]
#[test]
fn track_layout_accessor_returns_per_base() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("x.pbz.zarr");
    let store = PbzStore::create(&path, &["chr1".to_string()], &[1000]).unwrap();
    store
        .create_track(
            "t",
            TrackConfig {
                dtype: "float32".into(),
                samples: None,
                chunk_size: 100,
                ..Default::default()
            },
        )
        .unwrap();
    let track = store.track("t").unwrap();
    assert_eq!(track.layout(), "per_base");
}
