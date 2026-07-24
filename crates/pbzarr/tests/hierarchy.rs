//! Polymorphic open: a pbz artifact is a collection or a standalone track,
//! dispatched on `perbase:kind`.

use ndarray::{Array1, ArrayD};
use pbzarr::genome::Contig;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzNode, PbzStore, Track, TrackConfig};
use tempfile::TempDir;

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

/// Build a store with one `depth` track, write a small region, return the paths.
fn store_with_depth() -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let store_path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();
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
    let track_path = store_path.join("depth");
    (dir, store_path, track_path)
}

#[test]
fn pbznode_open_dispatches_collection_vs_track() {
    let (_dir, store_path, track_path) = store_with_depth();

    match PbzNode::open(&store_path).unwrap() {
        PbzNode::Collection(store) => {
            assert_eq!(store.track_names().collect::<Vec<_>>(), vec!["depth"]);
        }
        PbzNode::Track(_) => panic!("store root opened as a track"),
    }

    match PbzNode::open(&track_path).unwrap() {
        PbzNode::Track(t) => assert_eq!(t.dtype(), Dtype::I32),
        PbzNode::Collection(_) => panic!("track group opened as a collection"),
    }
}

#[test]
fn standalone_track_open_roundtrips() {
    let (_dir, _store_path, track_path) = store_with_depth();

    let track = Track::open(&track_path).unwrap();
    assert_eq!(track.name(), "depth"); // from the directory stem
    assert_eq!(track.dtype(), Dtype::I32);
    assert_eq!(track.total_len(), 1500);
    assert_eq!(track.genome().checksum(), tiny_genome().checksum());

    let region = track
        .genome()
        .resolve(&"chr2:100-110".parse().unwrap())
        .unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vec![42i32; 10]);
}

#[test]
fn open_guards_cross_kind() {
    let (_dir, store_path, track_path) = store_with_depth();

    // A track group is not a collection.
    assert!(PbzStore::open(&track_path).is_err());
    // A collection root is not a standalone track.
    assert!(Track::open(&store_path).is_err());
}
