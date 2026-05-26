use ndarray::{Array1, Array2};
use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};
use tempfile::TempDir;

#[test]
fn create_scalar_track_writes_per_contig_arrays_and_updates_root_metadata() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 1_000,
        },
        Contig {
            name: "chr2".into(),
            length: 500,
        },
    ])
    .unwrap();

    let mut store = PbzStore::create(&path, genome, None).unwrap();
    store
        .create_track("mask", TrackConfig::new(Dtype::Bool))
        .unwrap();

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

    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 1_000,
    }])
    .unwrap();

    let mut store = PbzStore::create(&path, genome, None).unwrap();
    let cfg = TrackConfig::new(Dtype::U16)
        .columns(vec!["A".into(), "B".into(), "C".into()])
        .column_dim("sample");
    store.create_track("depth", cfg).unwrap();

    drop(store);

    // per-contig data array exists, sample coord array exists
    assert!(path.join("chr1").join("depth").join("zarr.json").exists());
    assert!(path.join("chr1").join("sample").join("zarr.json").exists());

    // re-open and confirm root metadata records "depth" with column_dim "sample"
    let store = PbzStore::open(&path).unwrap();
    assert!(store.track_names().any(|n| n == "depth"));
}

#[test]
fn write_then_read_scalar_track_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 4_000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path, genome, None).unwrap();
    store
        .create_track("mask", TrackConfig::new(Dtype::Bool))
        .unwrap();

    let region = Region {
        contig: store.genome().id("chr1").unwrap(),
        start: 1_000,
        end: 1_500,
    };
    let mut data = Array1::from_elem(500, false);
    for i in 0..500 {
        if i % 3 == 0 {
            data[i] = true;
        }
    }
    let dyn_view = data.view().into_dyn();
    store
        .track("mask")
        .unwrap()
        .write_region(&region, dyn_view)
        .unwrap();

    let got = store
        .track("mask")
        .unwrap()
        .read_region::<bool>(&region)
        .unwrap();
    assert_eq!(got.shape(), &[500]);
    let got_flat: Array1<bool> = got.into_dimensionality::<ndarray::Ix1>().unwrap();
    assert_eq!(got_flat, data);
}

#[test]
fn write_then_read_cohort_track_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 4_000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path, genome, None).unwrap();
    store
        .create_track(
            "depth",
            TrackConfig::new(Dtype::U16)
                .columns(vec!["A".into(), "B".into(), "C".into()])
                .column_dim("sample"),
        )
        .unwrap();

    let region = Region {
        contig: store.genome().id("chr1").unwrap(),
        start: 0,
        end: 100,
    };
    let mut data = Array2::<u16>::zeros((100, 3));
    for i in 0..100 {
        data[[i, 0]] = i as u16;
        data[[i, 1]] = (i * 2) as u16;
        data[[i, 2]] = (i * 3) as u16;
    }
    store
        .track("depth")
        .unwrap()
        .write_region(&region, data.view().into_dyn())
        .unwrap();

    let got = store
        .track("depth")
        .unwrap()
        .read_region::<u16>(&region)
        .unwrap();
    let got2: Array2<u16> = got.into_dimensionality::<ndarray::Ix2>().unwrap();
    assert_eq!(got2, data);
}

#[test]
fn partial_chunk_write_then_read() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 10_000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path, genome, None).unwrap();
    // chunk_size 1_000; write a 500-bp region crossing chunk boundary at 1000
    let cfg = TrackConfig::new(Dtype::U32).chunk_size(1_000);
    store.create_track("x", cfg).unwrap();

    let region = Region {
        contig: store.genome().id("chr1").unwrap(),
        start: 800,
        end: 1_300,
    };
    let data = Array1::<u32>::from_iter(0..500u32);
    store
        .track("x")
        .unwrap()
        .write_region(&region, data.view().into_dyn())
        .unwrap();

    let got = store
        .track("x")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    let got1: Array1<u32> = got.into_dimensionality::<ndarray::Ix1>().unwrap();
    assert_eq!(got1, data);
}

#[test]
fn sharded_scalar_track_round_trips() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.pbz");
    let genome = Genome::new(vec![Contig { name: "chr1".into(), length: 10_000 }]).unwrap();
    let mut store = PbzStore::create(&path, genome, None).unwrap();

    // chunk_size 1000, shard_size 4000 (= 4 chunks per shard)
    store
        .create_track(
            "depth",
            TrackConfig::new(Dtype::U32)
                .chunk_size(1_000)
                .shard_size(4_000),
        )
        .unwrap();

    let region = pbzarr::Region {
        contig: store.genome().id("chr1").unwrap(),
        start: 0,
        end: 10_000,
    };
    let data = ndarray::Array1::<u32>::from_iter(0..10_000u32);
    store
        .track("depth")
        .unwrap()
        .write_region(&region, data.view().into_dyn())
        .unwrap();

    let got = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    let arr = got.into_dimensionality::<ndarray::Ix1>().unwrap();
    assert_eq!(arr, data);
}
