use ndarray::Array2;
use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::Import;
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, ExplicitArraySpec, TrackConfig};
use serde_json::json;
use tempfile::TempDir;

/// Column `column` of a `width`-wide track: position `p` reads as
/// `p * width + column`.
#[derive(Clone)]
struct GridReader {
    genome: Reference,
    schema: OutputSchema,
    column: u64,
    width: u64,
}

impl ValueReader for GridReader {
    fn contigs(&self) -> &Reference {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        _contig: &str,
        start: u64,
        _end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        let value = |row: usize| (start + row as u64) * self.width + self.column;
        match &mut outputs[0] {
            OutputSinkMut::I32(dst) => dst.iter_mut().enumerate().for_each(|(i, v)| {
                *v = value(i) as i32;
            }),
            OutputSinkMut::U16(dst) => dst.iter_mut().enumerate().for_each(|(i, v)| {
                *v = value(i) as u16;
            }),
            _ => panic!("test reader supports i32/u16 sinks only"),
        }
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(self.clone())
    }
}

fn genome(len: u64) -> Reference {
    Reference::new(vec![Contig {
        name: "chr1".into(),
        length: len,
    }])
    .unwrap()
}

/// Create a store at `path` and import one `width`-column track `depth`.
fn import(path: &std::path::Path, len: u64, cfg: TrackConfig) -> pbzarr::Result<Dataset> {
    let genome = genome(len);
    let width = cfg.columns.as_ref().map_or(1, |c| c.len() as u64);
    let mut ds = Dataset::create(path, genome.clone())?;
    let readers = (0..width)
        .map(|column| GridReader {
            genome: genome.clone(),
            schema: OutputSchema::single("value", cfg.dtype),
            column,
            width,
        })
        .collect();
    let columns = cfg.columns.is_some();
    let builder = Import::from_readers(readers)?.into_dataset(&mut ds, vec![("depth".into(), cfg)]);
    let builder = if columns {
        builder.readers_as_columns()
    } else {
        builder
    };
    builder.run()?;
    Ok(ds)
}

fn array_meta(store_path: &std::path::Path, track: &str) -> serde_json::Value {
    let path = store_path.join(track).join("zarr.json");
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
    let cfg = TrackConfig::new(Dtype::I32)
        .columns(vec!["a".into(), "b".into()])
        .chunk_size(4)
        .codecs(spec);
    let ds = import(&path, 8, cfg).unwrap();

    let data = Array2::from_shape_fn((8, 2), |(i, j)| (i * 2 + j) as i32).into_dyn();
    assert_eq!(ds.read_raw::<i32>("depth", &[0..8, 0..2]).unwrap(), data);

    let meta = array_meta(&path, "depth");
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
    let cfg = TrackConfig::new(Dtype::I32)
        .columns(vec!["a".into(), "b".into()])
        .chunk_size(4)
        .codecs(spec);
    let ds = import(&path, 8, cfg).unwrap();

    let data = Array2::from_shape_fn((8, 2), |(i, j)| (i * 2 + j) as i32).into_dyn();
    assert_eq!(ds.read_raw::<i32>("depth", &[0..8, 0..2]).unwrap(), data);

    // A chunk-unaligned read goes through the partial-decode chain, where the
    // pcodec full-decode cache must use the transposed representation.
    assert_eq!(
        ds.read_raw::<i32>("depth", &[1..6, 0..2]).unwrap(),
        data.slice(ndarray::s![1..6, ..]).to_owned().into_dyn()
    );

    let meta = array_meta(&path, "depth");
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
    let cfg = TrackConfig::new(Dtype::U16)
        .columns(vec!["a".into(), "b".into(), "c".into(), "d".into()])
        .codecs(spec);
    let ds = import(&path, 16, cfg).unwrap();

    let data = Array2::from_shape_fn((16, 4), |(i, j)| (i * 4 + j) as u16).into_dyn();
    assert_eq!(ds.read_raw::<u16>("depth", &[0..16, 0..4]).unwrap(), data);

    let meta = array_meta(&path, "depth");
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
    assert!(import(&path, 16, TrackConfig::new(Dtype::I32).codecs(spec)).is_err());
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
