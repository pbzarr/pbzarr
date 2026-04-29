use ndarray::{Array1, Array2};
use pbzarr::{PbzStore, TrackConfig};
use tempfile::TempDir;

#[test]
fn full_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let contigs = vec!["chr1".to_string(), "chr2".to_string()];
    let lengths = vec![5000u64, 3000];

    let store = PbzStore::create(&path, &contigs, &lengths).unwrap();

    let config = TrackConfig {
        dtype: "uint32".into(),
        columns: Some(vec!["s1".into(), "s2".into(), "s3".into()]),
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
    assert!(track2.has_columns());
    assert_eq!(track2.columns().unwrap(), &["s1", "s2", "s3"]);
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

#[test]
fn bool_scalar_track_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz.zarr");

    let store = PbzStore::create(&path, &["chr1".into()], &[2500]).unwrap();

    let config = TrackConfig {
        dtype: "bool".into(),
        columns: None,
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
    assert!(!track2.has_columns());

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
                columns: Some(vec!["s1".into()]),
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
                columns: None,
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
                columns: Some(vec!["pop1".into(), "pop2".into()]),
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
                columns: Some(vec!["s1".into()]),
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
                columns: Some(vec!["s1".into()]),
                chunk_size: 100,
                column_chunk_size: 1,
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
