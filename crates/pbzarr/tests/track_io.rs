use ndarray::{Array1, ArrayD};
use pbzarr::genome::Contig;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore, TrackConfig};
use tempfile::TempDir;

fn two_contig_store(path: &std::path::Path) -> PbzStore {
    let g = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 1000,
        },
        Contig {
            name: "chr2".into(),
            length: 500,
        },
    ])
    .unwrap();
    let mut store = PbzStore::create(path).unwrap();
    store
        .create_track("depth", g, TrackConfig::new(Dtype::I32))
        .unwrap();
    store
}

#[test]
fn write_and_read_region_translates_offset() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let store = two_contig_store(&path);
    let track = store.track("depth").unwrap();

    // Write [100,110) on chr2 (flat base = 1000).
    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr2:100-110".parse().unwrap())
        .unwrap();
    let data: ArrayD<i32> = Array1::from(vec![7i32; 10]).into_dyn();
    track.write_region(&region, data).unwrap();

    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vec![7i32; 10]);

    // The same flat positions on chr1 must be untouched (fill = 0).
    let chr1 = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:100-110".parse().unwrap())
        .unwrap();
    let other: ArrayD<i32> = track.read_region(&chr1).unwrap();
    assert_eq!(other.into_raw_vec_and_offset().0, vec![0i32; 10]);
}

#[test]
fn cohort_track_roundtrips_columns() {
    use ndarray::Array2;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 100,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            g,
            TrackConfig::new(Dtype::I32)
                .columns(vec!["a".into(), "b".into(), "c".into()])
                .column_dim("sample"),
        )
        .unwrap();
    let track = store.track("depth").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.column_dim(), Some("sample"));
    assert_eq!(track.columns_count().unwrap(), 3);

    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:0-4".parse().unwrap())
        .unwrap();
    let data: ArrayD<i32> = Array2::from_shape_vec((4, 3), (0..12).collect())
        .unwrap()
        .into_dyn();
    track.write_region(&region, data.clone()).unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got, data);
}

#[test]
fn cohort_track_reopens_with_column_dim() {
    use ndarray::Array2;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 100,
    }])
    .unwrap();
    let data: ArrayD<i32> = Array2::from_shape_vec((4, 3), (0..12).collect())
        .unwrap()
        .into_dyn();
    {
        let mut store = PbzStore::create(&path).unwrap();
        store
            .create_track(
                "depth",
                g,
                TrackConfig::new(Dtype::I32)
                    .columns(vec!["a".into(), "b".into(), "c".into()])
                    .column_dim("sample"),
            )
            .unwrap();
        let track = store.track("depth").unwrap();
        let region = store
            .genome_for("depth")
            .unwrap()
            .resolve(&"chr1:0-4".parse().unwrap())
            .unwrap();
        track.write_region(&region, data.clone()).unwrap();
    }

    let store = PbzStore::open(&path).unwrap();
    let track = store.track("depth").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.column_dim(), Some("sample"));
    assert_eq!(track.columns_count().unwrap(), 3);

    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:0-4".parse().unwrap())
        .unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got, data);
}

#[test]
fn region_spanning_chunk_boundary_is_correct() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 30,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    // chunk_size 10 → region [5,25) spans three chunks.
    store
        .create_track("d", g, TrackConfig::new(Dtype::I32).chunk_size(10))
        .unwrap();
    let track = store.track("d").unwrap();
    let region = store
        .genome_for("d")
        .unwrap()
        .resolve(&"chr1:5-25".parse().unwrap())
        .unwrap();
    let vals: Vec<i32> = (0..20).collect();
    track
        .write_region(&region, Array1::from(vals.clone()).into_dyn())
        .unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vals);
}
