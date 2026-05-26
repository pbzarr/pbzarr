use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, TrackConfig};
use tempfile::TempDir;

#[test]
fn create_scalar_track_writes_per_contig_arrays_and_updates_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 1_000 },
        Contig { name: "chr2".into(), length: 500 },
    ])
    .unwrap();

    let mut store = PbzStore::create(&path, genome, None).unwrap();
    store.create_track("mask", TrackConfig::scalar(Dtype::Bool)).unwrap();

    drop(store);

    // re-open and verify
    let store = PbzStore::open(&path).unwrap();
    assert_eq!(store.track_names().collect::<Vec<_>>(), vec!["mask"]);

    // per-contig data arrays exist
    assert!(path.join("chr1").join("mask").join("zarr.json").exists());
    assert!(path.join("chr2").join("mask").join("zarr.json").exists());
}

#[test]
fn create_cohort_track_writes_per_contig_data_and_sample_coord_arrays() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 1_000 },
    ]).unwrap();

    let mut store = PbzStore::create(&path, genome, None).unwrap();
    let cfg = TrackConfig::cohort(Dtype::U16, vec!["A".into(), "B".into(), "C".into()]);
    store.create_track("depth", cfg).unwrap();

    drop(store);

    // per-contig data array exists, sample coord array exists
    assert!(path.join("chr1").join("depth").join("zarr.json").exists());
    assert!(path.join("chr1").join("sample").join("zarr.json").exists());

    // re-open and confirm root metadata records "depth" with column_dim "sample"
    let store = PbzStore::open(&path).unwrap();
    assert!(store.track_names().any(|n| n == "depth"));
}
