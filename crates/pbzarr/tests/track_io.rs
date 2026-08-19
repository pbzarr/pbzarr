use std::sync::Arc;

use ndarray::{Array1, ArrayD};
use pbzarr::genome::{Contig, Region};
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore, TrackConfig};
use tempfile::TempDir;
use zarrs::array::{ArrayBuilder, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

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
fn read_region_past_contig_end_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let store = two_contig_store(&path);
    let track = store.track("depth").unwrap();
    let genome = store.genome_for("depth").unwrap();

    // Put known data at chr2's start (flat positions 1000..1010).
    let chr2 = genome.resolve(&"chr2:0-10".parse().unwrap()).unwrap();
    track
        .write_region(&chr2, Array1::from(vec![7i32; 10]).into_dyn())
        .unwrap();

    // chr1 has length 1000; [995, 1005) overhangs into chr2's flat range.
    let region = Region {
        contig: genome.id("chr1").unwrap(),
        start: 995,
        end: 1005,
    };
    let err = track
        .read_region::<i32>(&region)
        .expect_err("read past contig end must error, not return chr2's values")
        .to_string();
    assert!(err.contains("chr1"), "{err}");
    assert!(err.contains("1000"), "{err}");
}

#[test]
fn write_region_past_contig_end_errors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let store = two_contig_store(&path);
    let track = store.track("depth").unwrap();
    let genome = store.genome_for("depth").unwrap();

    let region = Region {
        contig: genome.id("chr1").unwrap(),
        start: 995,
        end: 1005,
    };
    let err = track
        .write_region(&region, Array1::from(vec![9i32; 10]).into_dyn())
        .expect_err("write past contig end must error, not spill into chr2")
        .to_string();
    assert!(err.contains("chr1"), "{err}");
    assert!(err.contains("1000"), "{err}");

    // chr2's start must be untouched.
    let chr2 = genome.resolve(&"chr2:0-5".parse().unwrap()).unwrap();
    let got: ArrayD<i32> = track.read_region(&chr2).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vec![0i32; 5]);
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
    assert_eq!(
        track.column_labels().unwrap(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
    );

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
fn reopened_track_reads_column_labels_across_coordinate_chunks() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multi_chunk_labels.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    {
        let mut store = PbzStore::create(&path).unwrap();
        store
            .create_track(
                "depth",
                g,
                TrackConfig::new(Dtype::I32)
                    .columns(vec![
                        "a".into(),
                        "b".into(),
                        "c".into(),
                        "d".into(),
                        "e".into(),
                    ])
                    .column_dim("sample"),
            )
            .unwrap();
    }

    let storage: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
    let labels = ArrayBuilder::new(vec![5], vec![2], data_type::string(), "")
        .dimension_names(["sample"].into())
        .build(storage, "/depth/sample")
        .unwrap();
    labels.store_metadata().unwrap();
    labels
        .store_chunk(
            &[0],
            Array1::from(vec!["a".to_owned(), "b".to_owned()]).into_dyn(),
        )
        .unwrap();
    labels
        .store_chunk(
            &[1],
            Array1::from(vec!["c".to_owned(), "d".to_owned()]).into_dyn(),
        )
        .unwrap();
    labels
        .store_chunk(
            &[2],
            Array1::from(vec!["e".to_owned(), String::new()]).into_dyn(),
        )
        .unwrap();

    let store = PbzStore::open(&path).unwrap();
    assert_eq!(
        store.track("depth").unwrap().column_labels().unwrap(),
        vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
            "e".to_owned()
        ]
    );
}

#[test]
fn column_labels_rejects_a_rank_zero_coordinate_array() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("rank_zero_labels.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    {
        let mut store = PbzStore::create(&path).unwrap();
        store
            .create_track(
                "depth",
                g,
                TrackConfig::new(Dtype::I32)
                    .columns(vec!["a".into(), "b".into()])
                    .column_dim("sample"),
            )
            .unwrap();
    }
    let storage: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
    ArrayBuilder::new(
        Vec::<u64>::new(),
        Vec::<u64>::new(),
        data_type::string(),
        "",
    )
    .build(storage, "/depth/sample")
    .unwrap()
    .store_metadata()
    .unwrap();

    let store = PbzStore::open(&path).unwrap();
    let error = store
        .track("depth")
        .unwrap()
        .column_labels()
        .expect_err("rank-zero column coordinates must return an error")
        .to_string();
    assert!(error.contains("rank 1"), "{error}");
}

#[test]
fn column_labels_rejects_a_coordinate_width_different_from_values() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("wrong_width_labels.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    {
        let mut store = PbzStore::create(&path).unwrap();
        store
            .create_track(
                "depth",
                g,
                TrackConfig::new(Dtype::I32)
                    .columns(vec!["a".into(), "b".into()])
                    .column_dim("sample"),
            )
            .unwrap();
    }
    let storage: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
    let labels = ArrayBuilder::new(vec![1], vec![1], data_type::string(), "")
        .dimension_names(["sample"].into())
        .build(storage, "/depth/sample")
        .unwrap();
    labels.store_metadata().unwrap();
    labels
        .store_chunk(&[0], Array1::from(vec!["a".to_owned()]).into_dyn())
        .unwrap();

    let store = PbzStore::open(&path).unwrap();
    let error = store
        .track("depth")
        .unwrap()
        .column_labels()
        .expect_err("column coordinate width must match values")
        .to_string();
    assert!(error.contains("1 labels for 2 columns"), "{error}");
}

#[test]
fn column_labels_rejects_coordinate_dimension_names_that_do_not_match_the_track() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("wrong_dimension_name_labels.pbz");
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    {
        let mut store = PbzStore::create(&path).unwrap();
        store
            .create_track(
                "depth",
                g,
                TrackConfig::new(Dtype::I32)
                    .columns(vec!["a".into(), "b".into()])
                    .column_dim("sample"),
            )
            .unwrap();
    }
    let storage: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
    let labels = ArrayBuilder::new(vec![2], vec![2], data_type::string(), "")
        .dimension_names(["wrong"].into())
        .build(storage, "/depth/sample")
        .unwrap();
    labels.store_metadata().unwrap();
    labels
        .store_chunk(
            &[0],
            Array1::from(vec!["a".to_owned(), "b".to_owned()]).into_dyn(),
        )
        .unwrap();

    let store = PbzStore::open(&path).unwrap();
    let error = store
        .track("depth")
        .unwrap()
        .column_labels()
        .expect_err("coordinate dimension_names must match the track column dimension")
        .to_string();
    assert!(error.contains("dimension_names"), "{error}");
    assert!(error.contains("wrong"), "{error}");
    assert!(error.contains("sample"), "{error}");
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
fn sharded_cohort_track_roundtrips_across_shard_boundary() {
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
                .columns(vec!["a".into(), "b".into()])
                .chunk_size(10)
                .column_chunk_size(2)
                .shard_size(40),
        )
        .unwrap();
    // chunk_size() reports the shard extent, not the subchunk.
    let track = store.track("depth").unwrap();
    assert_eq!(track.chunk_size().unwrap(), 40);

    // Region [30,55) straddles the shard boundary at 40.
    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:30-55".parse().unwrap())
        .unwrap();
    let data: ArrayD<i32> = Array2::from_shape_vec((25, 2), (0..50).collect())
        .unwrap()
        .into_dyn();
    track.write_region(&region, data.clone()).unwrap();

    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got, data);

    let store = PbzStore::open(&path).unwrap();
    let track = store.track("depth").unwrap();
    assert_eq!(track.chunk_size().unwrap(), 40);
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got, data);
}

#[test]
fn custom_fill_value_fills_untouched_positions() {
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
            "d",
            g,
            TrackConfig::new(Dtype::I32).fill_value(serde_json::json!(-1)),
        )
        .unwrap();
    let track = store.track("d").unwrap();
    // Never written → reads back the custom fill, not the dtype default of 0.
    let region = store
        .genome_for("d")
        .unwrap()
        .resolve(&"chr1:0-10".parse().unwrap())
        .unwrap();
    let got: ArrayD<i32> = track.read_region(&region).unwrap();
    assert_eq!(got.into_raw_vec_and_offset().0, vec![-1i32; 10]);
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
