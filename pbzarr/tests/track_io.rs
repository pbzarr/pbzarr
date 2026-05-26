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
