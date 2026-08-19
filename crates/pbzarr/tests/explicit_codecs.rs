use ndarray::Array2;
use pbzarr::io::Dtype;
use pbzarr::{Contig, ExplicitArraySpec, Genome, PbzStore, TrackConfig};
use serde_json::json;
use tempfile::TempDir;

fn genome(len: u64) -> Genome {
    Genome::new(vec![Contig {
        name: "chr1".into(),
        length: len,
    }])
    .unwrap()
}

fn values_meta(store_path: &std::path::Path, track: &str) -> serde_json::Value {
    let path = store_path.join(track).join("values").join("zarr.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn codec_names(meta: &serde_json::Value) -> Vec<String> {
    meta["codecs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn explicit_codecs_array_transpose_zstd_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.pbz");
    let spec = ExplicitArraySpec::parse(
        r#"[{"name":"transpose","configuration":{"order":[1,0]}},
            {"name":"zstd","configuration":{"level":3,"checksum":false}}]"#,
    )
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome(8),
            TrackConfig::new(Dtype::I32)
                .columns(vec!["a".into(), "b".into()])
                .chunk_size(4)
                .codecs(spec),
        )
        .unwrap();

    let track = store.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-8".parse().unwrap())
        .unwrap();
    let data = Array2::from_shape_fn((8, 2), |(i, j)| (i * 2 + j) as i32).into_dyn();
    track.write_region(&region, data.clone()).unwrap();
    assert_eq!(track.read_region::<i32>(&region).unwrap(), data);

    let meta = values_meta(&path, "depth");
    assert_eq!(codec_names(&meta), ["transpose", "bytes", "zstd"]);
    assert_eq!(meta["codecs"][0]["configuration"]["order"], json!([1, 0]));
}

#[test]
fn explicit_codecs_transpose_pcodec_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("p.pbz");
    let spec = ExplicitArraySpec::parse(
        r#"[{"name":"transpose","configuration":{"order":[1,0]}},
            {"name":"numcodecs.pcodec","configuration":{"level":8}}]"#,
    )
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome(8),
            TrackConfig::new(Dtype::I32)
                .columns(vec!["a".into(), "b".into()])
                .chunk_size(4)
                .codecs(spec),
        )
        .unwrap();

    let track = store.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-8".parse().unwrap())
        .unwrap();
    let data = Array2::from_shape_fn((8, 2), |(i, j)| (i * 2 + j) as i32).into_dyn();
    track.write_region(&region, data.clone()).unwrap();
    assert_eq!(track.read_region::<i32>(&region).unwrap(), data);

    let meta = values_meta(&path, "depth");
    assert_eq!(codec_names(&meta), ["transpose", "numcodecs.pcodec"]);
}

#[test]
fn explicit_fragment_controls_grid_and_sharding() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let spec = ExplicitArraySpec::parse(
        r#"{
          "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 4]}},
          "codecs": [{
            "name": "sharding_indexed",
            "configuration": {
              "chunk_shape": [4, 2],
              "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                         {"name": "zstd", "configuration": {"level": 3, "checksum": false}}],
              "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                               {"name": "crc32c"}],
              "index_location": "end"
            }
          }]
        }"#,
    )
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome(16),
            TrackConfig::new(Dtype::U16)
                .columns(vec!["a".into(), "b".into(), "c".into(), "d".into()])
                .codecs(spec),
        )
        .unwrap();

    let track = store.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-16".parse().unwrap())
        .unwrap();
    let data = Array2::from_shape_fn((16, 4), |(i, j)| (i * 4 + j) as u16).into_dyn();
    track.write_region(&region, data.clone()).unwrap();
    assert_eq!(track.read_region::<u16>(&region).unwrap(), data);
    assert_eq!(track.chunk_size().unwrap(), 8);

    let meta = values_meta(&path, "depth");
    assert_eq!(
        meta["chunk_grid"]["configuration"]["chunk_shape"],
        json!([8, 4])
    );
    assert_eq!(codec_names(&meta), ["sharding_indexed"]);
}

#[test]
fn chunk_grid_rank_mismatch_errors_at_creation() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("bad.pbz");
    let spec = ExplicitArraySpec::parse(
        r#"{"chunk_grid": {"name":"regular","configuration":{"chunk_shape":[8,4]}},
            "codecs": [{"name":"zstd","configuration":{"level":3,"checksum":false}}]}"#,
    )
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    let result = store.create_track(
        "depth",
        genome(16),
        TrackConfig::new(Dtype::I32).codecs(spec),
    );
    assert!(result.is_err());
}

#[test]
fn import_config_threads_codecs_into_track_config() {
    let spec = ExplicitArraySpec::parse(
        r#"[{"name":"zstd","configuration":{"level":3,"checksum":false}}]"#,
    )
    .unwrap();
    let config = pbzarr::import::Config {
        codecs: Some(spec),
        ..Default::default()
    };
    let track_config = config.track_config(Dtype::I32, None, None, "sample");
    assert!(track_config.codecs.is_some());
}
