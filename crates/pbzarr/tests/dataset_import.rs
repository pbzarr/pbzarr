// One range per axis; single-range selections are intentionally rank one.
#![allow(clippy::single_range_in_vec_init)]
use pbzarr::Contig;
use pbzarr::dataset::{Dataset, Reference, Role, ValidationStatus};
use std::sync::Arc;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

fn reference() -> Reference {
    Reference::new(vec![
        Contig {
            name: "chr1".into(),
            length: 5,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap()
}
#[test]
fn creates_shared_reference() {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let ds = Dataset::create_with_storage(storage.clone(), reference()).unwrap();
    assert_eq!(ds.reference().unwrap().contigs(), reference().contigs());
    assert_eq!(ds.variables()["contig_length"].role, Role::Coordinate);
    assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    let reopened = Dataset::open_with_storage(storage).unwrap();
    assert_eq!(
        reopened
            .read_raw::<i64>("contig_length", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[5, 7]
    );
}
#[test]
fn creates_empty_and_zero_length_references() {
    for contigs in [
        vec![],
        vec![Contig {
            name: "empty".into(),
            length: 0,
        }],
    ] {
        let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let ds = Dataset::create_with_storage(storage, Reference::new(contigs.clone()).unwrap())
            .unwrap();
        assert_eq!(ds.reference().unwrap().contigs(), contigs);
        assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    }
}
#[test]
fn creation_rejects_occupied_namespaces_without_writing() {
    for key in ["orphan/c/0", "bare", "zarr.json"] {
        let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let key = StoreKey::new(key).unwrap();
        storage.set(&key, b"untouched".to_vec().into()).unwrap();
        assert!(Dataset::create_with_storage(storage.clone(), reference()).is_err());
        assert_eq!(storage.get(&key).unwrap().unwrap().as_ref(), b"untouched");
        assert!(
            storage
                .get(&StoreKey::new("contig/zarr.json").unwrap())
                .unwrap()
                .is_none()
        );
    }
}

use pbzarr::import::{Import, PipelineOptions};
use pbzarr::io::{Dtype, OutputField, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{ExplicitArraySpec, Genome, PbzStore, TrackConfig};
use std::sync::Mutex;
use zarrs::storage::storage_adapter::usage_log::UsageLogStorageAdapter;

#[derive(Clone)]
struct ConstReader {
    genome: Genome,
    schema: OutputSchema,
    values: Vec<u32>,
}
impl ConstReader {
    fn new(values: Vec<u32>) -> Self {
        let schema = OutputSchema::new(
            values
                .iter()
                .enumerate()
                .map(|(i, _)| OutputField {
                    name: Some(format!("field_{i}")),
                    dtype: Dtype::U32,
                })
                .collect(),
        );
        Self {
            genome: Genome::new(reference().contigs().to_vec()).unwrap(),
            schema,
            values,
        }
    }
}
impl ValueReader for ConstReader {
    fn contigs(&self) -> &Genome {
        &self.genome
    }
    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }
    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(self.clone())
    }
    fn read_into(
        &mut self,
        _: &str,
        _: u64,
        _: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        for (out, value) in outputs.iter_mut().zip(&self.values) {
            out.as_u32_mut()?.fill(*value);
        }
        Ok(())
    }
}
type Trace = Arc<Mutex<Vec<u8>>>;
fn traced_dataset() -> (Dataset, ReadableWritableListableStorage, Trace) {
    let trace = Arc::new(Mutex::new(Vec::new()));
    let storage: ReadableWritableListableStorage = Arc::new(UsageLogStorageAdapter::new(
        Arc::new(MemoryStore::new()),
        trace.clone(),
        String::new,
    ));
    let ds = Dataset::create_with_storage(storage.clone(), reference()).unwrap();
    trace.lock().unwrap().clear();
    (ds, storage, trace)
}
fn no_mutations(trace: &Trace) {
    let trace = String::from_utf8(trace.lock().unwrap().clone()).unwrap();
    assert!(
        !trace.contains("set(") && !trace.contains("set_partial") && !trace.contains("erase"),
        "{trace}"
    );
}
#[test]
fn singleton_import_and_estimate_use_root_arrays() {
    let (mut ds, storage, trace) = traced_dataset();
    let cfg = TrackConfig::new(Dtype::U32)
        .columns(vec!["S1".into()])
        .column_dim("sample")
        .chunk_size(4);
    let builder = Import::from_readers(vec![ConstReader::new(vec![7])])
        .unwrap()
        .into_dataset(&mut ds, vec![("depth".into(), cfg)])
        .readers_as_columns()
        .expect_column_labels(vec!["S1".into()]);
    builder.estimate().unwrap();
    no_mutations(&trace);
    builder.run().unwrap();
    assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
    assert_eq!(ds.variables()["depth"].shape, [12, 1]);
    assert_eq!(
        ds.read_raw::<u32>("depth", &[0..12, 0..1]).unwrap(),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[12, 1]), 7)
    );
    assert!(
        storage
            .get(&StoreKey::new("depth/values/zarr.json").unwrap())
            .unwrap()
            .is_none()
    );
    assert_eq!(ds.variables()["sample"].role, Role::Coordinate);
    let trace = String::from_utf8(trace.lock().unwrap().clone()).unwrap();
    let last_chunk = trace.rfind("set(depth/c/").unwrap();
    let metadata = trace.find("set(depth/zarr.json").unwrap();
    assert!(last_chunk < metadata, "{trace}");
}
#[test]
fn multifield_cohorts_preserve_shards_and_reuse_shared_labels() {
    for shard in [None, Some(8), Some(4)] {
        let (mut ds, storage, trace) = traced_dataset();
        let mut cfg = TrackConfig::new(Dtype::U32)
            .chunk_size(4)
            .columns(vec!["S1".into(), "S2".into(), "S3".into()])
            .column_dim("sample")
            .column_chunk_size(2);
        if let Some(size) = shard {
            cfg = cfg.shard_size(size).shard_column_size(2);
        }
        let readers = vec![
            ConstReader::new(vec![7, 11]),
            ConstReader::new(vec![17, 21]),
            ConstReader::new(vec![27, 31]),
        ];
        Import::from_readers(readers)
            .unwrap()
            .into_dataset(
                &mut ds,
                vec![("depth".into(), cfg.clone()), ("other".into(), cfg.clone())],
            )
            .fields_as_tracks()
            .readers_as_columns()
            .options(PipelineOptions {
                workers: 2,
                ..Default::default()
            })
            .run()
            .unwrap();
        for (name, expected) in [("depth", [7, 17, 27]), ("other", [11, 21, 31])] {
            let got = ds.read_raw::<u32>(name, &[0..12, 0..3]).unwrap();
            for row in got.outer_iter() {
                assert_eq!(row.as_slice().unwrap(), &expected);
            }
        }
        let labels = storage
            .get(&StoreKey::new("sample/zarr.json").unwrap())
            .unwrap()
            .unwrap();
        trace.lock().unwrap().clear();
        Import::from_readers(vec![
            ConstReader::new(vec![3]),
            ConstReader::new(vec![4]),
            ConstReader::new(vec![5]),
        ])
        .unwrap()
        .into_dataset(&mut ds, vec![("added".into(), cfg)])
        .readers_as_columns()
        .run()
        .unwrap();
        assert_eq!(
            storage
                .get(&StoreKey::new("sample/zarr.json").unwrap())
                .unwrap()
                .unwrap(),
            labels
        );
        let trace = String::from_utf8(trace.lock().unwrap().clone()).unwrap();
        assert!(!trace.contains("set(sample/"), "{trace}");
        assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    }
}
#[test]
fn one_dimensional_import_matches_legacy_and_elides_fill_chunks() {
    for value in [0, 19] {
        let (mut ds, storage, _) = traced_dataset();
        let cfg = TrackConfig::new(Dtype::U32)
            .chunk_size(4)
            .fill_value(serde_json::json!(0));
        Import::from_readers(vec![ConstReader::new(vec![value])])
            .unwrap()
            .into_dataset(&mut ds, vec![("depth".into(), cfg.clone())])
            .run()
            .unwrap();
        let got = ds.read_raw::<u32>("depth", &[0..12]).unwrap();
        assert_eq!(
            got,
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[12]), value)
        );
        let legacy_storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let mut legacy = PbzStore::create_with_storage(legacy_storage).unwrap();
        let genome = Genome::new(reference().contigs().to_vec()).unwrap();
        legacy.create_track("depth", genome.clone(), cfg).unwrap();
        Import::from_readers(vec![ConstReader::new(vec![value])])
            .unwrap()
            .into_track(legacy.track("depth").unwrap())
            .run()
            .unwrap();
        let old = legacy
            .track("depth")
            .unwrap()
            .read_region::<u32>(&pbzarr::Region {
                contig: genome.id("chr2").unwrap(),
                start: 0,
                end: 7,
            })
            .unwrap();
        assert_eq!(old.as_slice().unwrap(), &got.as_slice().unwrap()[5..]);
        if value == 0 {
            assert!(
                storage
                    .get(&StoreKey::new("depth/c/0").unwrap())
                    .unwrap()
                    .is_none()
            );
        }
    }
}
#[test]
fn invalid_imports_fail_before_any_mutation() {
    for case in [
        "reference",
        "dtype",
        "duplicate-label",
        "reserved-dim",
        "name-collision",
        "orphan",
        "bare",
        "codec",
        "grid",
        "duplicate-track",
        "shared-grid",
        "shard-shape",
    ] {
        let (mut ds, storage, trace) = traced_dataset();
        let mut reader = ConstReader::new(vec![7]);
        let mut cfg = TrackConfig::new(Dtype::U32).chunk_size(4);
        let mut name = "depth".to_string();
        match case {
            "reference" => { let mut contigs = reference().contigs().to_vec(); contigs.reverse(); reader.genome = Genome::new(contigs).unwrap(); }
            "dtype" => cfg.dtype = Dtype::I32,
            "shard-shape" => cfg = cfg.shard_size(6).columns(vec!["S1".into()]).column_dim("sample"),
            "duplicate-label" => cfg = cfg.columns(vec!["S1".into(), "S1".into()]),
            "reserved-dim" => cfg = cfg.columns(vec!["S1".into()]).column_dim("bin_4"),
            "name-collision" => name = "contig".into(),
            "orphan" | "bare" => { let key = if case == "orphan" {"depth/c/0"} else {"depth"};
                storage.set(&StoreKey::new(key).unwrap(), b"existing".to_vec().into()).unwrap(); },
            "codec" => cfg.codecs = Some(ExplicitArraySpec::parse(r#"[{"name":"not-a-codec"}]"#).unwrap()),
            "grid" => cfg.codecs = Some(ExplicitArraySpec::parse(r#"{"chunk_grid":{"name":"rectilinear","configuration":{"chunk_shape":[[4,8]]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#).unwrap()),
            _ => {}
        }
        let mut tracks = vec![(name, cfg.clone())];
        if case == "duplicate-track" {
            tracks.push(("depth".into(), cfg.clone()));
            reader = ConstReader::new(vec![7, 8]);
        }
        if case == "shared-grid" {
            tracks.push(("other".into(), cfg.clone().chunk_size(3)));
            reader = ConstReader::new(vec![7, 8]);
        }
        trace.lock().unwrap().clear();
        let fields = tracks.len() > 1;
        let columns = cfg.columns.is_some();
        let mut b = Import::from_readers(vec![reader])
            .unwrap()
            .into_dataset(&mut ds, tracks);
        if fields {
            b = b.fields_as_tracks();
        }
        if columns {
            b = b.readers_as_columns();
        }
        assert!(b.run().is_err(), "{case}");
        no_mutations(&trace);
    }
}
#[test]
fn shared_label_reordering_is_rejected_without_changes() {
    let (mut ds, _, trace) = traced_dataset();
    let cfg = TrackConfig::new(Dtype::U32)
        .columns(vec!["S1".into(), "S2".into()])
        .column_dim("sample");
    Import::from_readers(vec![ConstReader::new(vec![1]), ConstReader::new(vec![2])])
        .unwrap()
        .into_dataset(&mut ds, vec![("first".into(), cfg.clone())])
        .readers_as_columns()
        .run()
        .unwrap();
    trace.lock().unwrap().clear();
    let reversed = cfg.columns(vec!["S2".into(), "S1".into()]);
    assert!(
        Import::from_readers(vec![ConstReader::new(vec![3]), ConstReader::new(vec![4])])
            .unwrap()
            .into_dataset(&mut ds, vec![("second".into(), reversed)])
            .readers_as_columns()
            .run()
            .is_err()
    );
    no_mutations(&trace);
}

#[test]
fn explicit_codecs_and_notes_survive_root_import() {
    let (mut ds, _, _) = traced_dataset();
    let spec = ExplicitArraySpec::parse(
        r#"{
        "chunk_grid":{"name":"regular","configuration":{"chunk_shape":[4,1]}},
        "codecs":[{"name":"bytes","configuration":{"endian":"little"}},
                  {"name":"zstd","configuration":{"level":1,"checksum":false}}]}"#,
    )
    .unwrap();
    let mut cfg = TrackConfig::new(Dtype::U32)
        .chunk_size(1)
        .shard_size(3)
        .columns(vec!["S1".into()])
        .column_dim("sample")
        .description("Depth")
        .source("fixture")
        .codecs(spec);
    cfg.extra
        .insert("notes".into(), serde_json::json!("manual annotation"));
    Import::from_readers(vec![ConstReader::new(vec![7])])
        .unwrap()
        .into_dataset(&mut ds, vec![("depth".into(), cfg)])
        .readers_as_columns()
        .run()
        .unwrap();
    let meta = &ds.variables()["depth"].metadata;
    assert_eq!(
        meta["chunk_grid"]["configuration"]["chunk_shape"],
        serde_json::json!([4, 1])
    );
    assert_eq!(meta["codecs"][1]["name"], "zstd");
    assert_eq!(meta["codecs"][1]["configuration"]["level"], 1);
    assert_eq!(meta["attributes"]["description"], "Depth");
    assert_eq!(meta["attributes"]["source"], "fixture");
    assert_eq!(meta["attributes"]["notes"], "manual annotation");
    assert!(
        ds.read_raw::<u32>("depth", &[0..12, 0..1])
            .unwrap()
            .iter()
            .all(|&n| n == 7)
    );
}
#[test]
fn unsupported_neighbor_does_not_block_supported_import() {
    let (ds, storage, _) = traced_dataset();
    let mut metadata = ds.variables()["contig_length"].metadata.clone();
    metadata["codecs"] = serde_json::json!([{"name":"unavailable-codec"}]);
    storage
        .set(
            &StoreKey::new("foreign/zarr.json").unwrap(),
            serde_json::to_vec(&metadata).unwrap().into(),
        )
        .unwrap();
    let mut ds = Dataset::open_with_storage(storage).unwrap();
    Import::from_readers(vec![ConstReader::new(vec![7])])
        .unwrap()
        .into_dataset(
            &mut ds,
            vec![("depth".into(), TrackConfig::new(Dtype::U32))],
        )
        .run()
        .unwrap();
    assert!(ds.variables().contains_key("foreign"));
    assert!(
        ds.read_raw::<u32>("depth", &[0..12])
            .unwrap()
            .iter()
            .all(|&n| n == 7)
    );
}

#[test]
fn a_sharded_cohort_shards_one_column_chunk_at_any_width() {
    let (mut ds, _, _) = traced_dataset();
    let labels: Vec<String> = (0..5).map(|i| format!("S{i}")).collect();
    let cfg = TrackConfig::new(Dtype::U32)
        .columns(labels.clone())
        .column_dim("sample")
        .chunk_size(4)
        .column_chunk_size(2)
        .shard_size(8);
    let readers: Vec<ConstReader> = (0..5).map(|_| ConstReader::new(vec![7])).collect();
    Import::from_readers(readers)
        .unwrap()
        .into_dataset(&mut ds, vec![("depth".into(), cfg)])
        .readers_as_columns()
        .expect_column_labels(labels)
        .run()
        .unwrap();
    assert_eq!(ds.variables()["depth"].shape, [12, 5]);
    let shard = &ds.variables()["depth"].metadata["chunk_grid"]["configuration"]["chunk_shape"];
    // The column shard is one column chunk wide, so widening the column axis
    // later appends whole shards. A width of 5 is not a multiple of 2.
    assert_eq!(shard[1], 2);
    assert_eq!(
        ds.read_raw::<u32>("depth", &[0..12, 0..5]).unwrap(),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[12, 5]), 7)
    );
}
