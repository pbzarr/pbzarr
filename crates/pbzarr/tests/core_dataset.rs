use pbzarr::PbzError;
use pbzarr::dataset::{Dataset, FORMAT_VERSION, Role, Support};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use zarrs::storage::storage_adapter::usage_log::UsageLogStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

fn root(names: &str, lengths: &str) -> Value {
    json!({"zarr_format":3,"node_type":"group","attributes":{
        "zarr_conventions":[{"uuid":"b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c","name":"pbz:"}],
        "pbz:version":FORMAT_VERSION,"pbz:reference":{"names":names,"lengths":lengths},
        "coordinates":lengths}})
}
fn metadata(shape: &[u64], dims: &[&str], dtype: &str) -> Value {
    let chunk: Vec<u64> = shape.iter().map(|&n| n.max(1)).collect();
    json!({"zarr_format":3,"node_type":"array","shape":shape,"dimension_names":dims,
        "data_type":dtype,"chunk_grid":{"name":"regular","configuration":{"chunk_shape":chunk}},
        "chunk_key_encoding":{"name":"default","configuration":{"separator":"/"}},
        "fill_value":if dtype == "string" { json!("") } else { json!(0) },
        "codecs": if dtype == "string" {json!([{"name":"vlen-utf8"}])}
            else {json!([{"name":"bytes","configuration":{"endian":"little"}}])},
        "attributes":{}})
}
fn put(storage: &ReadableWritableListableStorage, key: &str, value: Value) {
    storage
        .set(
            &StoreKey::new(key).unwrap(),
            serde_json::to_vec(&value).unwrap().into(),
        )
        .unwrap();
}
type Trace = Arc<Mutex<Vec<u8>>>;
fn catalog_fixture(names: &str, lengths: &str) -> (ReadableWritableListableStorage, Trace) {
    let log = Arc::new(Mutex::new(Vec::<u8>::new()));
    let storage: ReadableWritableListableStorage = Arc::new(UsageLogStorageAdapter::new(
        Arc::new(MemoryStore::new()),
        log.clone(),
        String::new,
    ));
    put(&storage, "zarr.json", root(names, lengths));
    put(
        &storage,
        &format!("{names}/zarr.json"),
        metadata(&[2], &[names], "string"),
    );
    put(
        &storage,
        &format!("{lengths}/zarr.json"),
        metadata(&[2], &[names], "int64"),
    );
    put(
        &storage,
        "sample/zarr.json",
        metadata(&[3], &["sample"], "string"),
    );
    put(
        &storage,
        "depth/zarr.json",
        metadata(&[12, 3], &["position", "sample"], "uint16"),
    );
    log.lock().unwrap().clear();
    (storage, log)
}
#[test]
fn catalogs_native_coordinates_without_payload_reads() {
    for (names, lengths) in [("contig", "contig_length"), ("sequence", "sequence_length")] {
        let (storage, log) = catalog_fixture(names, lengths);
        let ds = Dataset::open_with_storage(storage).unwrap();
        assert_eq!(ds.variables()["depth"].role, Role::Measurement);
        assert_eq!(ds.variables()[names].role, Role::Coordinate);
        assert_eq!(ds.variables()[lengths].role, Role::Coordinate);
        assert_eq!(ds.variables()["depth"].dims, ["position", "sample"]);
        let trace = String::from_utf8(log.lock().unwrap().clone()).unwrap();
        assert!(!trace.contains("/c/"), "{trace}");
    }
}
#[test]
fn catalogs_unsupported_neighbors_scalars_auxiliary_coordinates_and_skips_groups() {
    let (storage, log) = catalog_fixture("contig", "contig_length");
    let mut unknown = metadata(&[12], &["position"], "uint16");
    unknown["codecs"] = json!([{"name":"unavailable-codec"}]);
    put(&storage, "unsupported/zarr.json", unknown);
    let mut scalar = metadata(&[], &[], "uint64");
    scalar.as_object_mut().unwrap().remove("dimension_names");
    put(&storage, "total/zarr.json", scalar);
    let mut depth = metadata(&[12, 3], &["position", "sample"], "uint16");
    depth["attributes"]["coordinates"] = json!("quality");
    put(&storage, "depth/zarr.json", depth);
    put(
        &storage,
        "quality/zarr.json",
        metadata(&[12], &["position"], "uint8"),
    );
    put(
        &storage,
        "extension/zarr.json",
        json!({"zarr_format":3,"node_type":"group","attributes":{}}),
    );
    put(
        &storage,
        "extension/hidden/zarr.json",
        metadata(&[1], &["x"], "uint8"),
    );
    log.lock().unwrap().clear();
    let ds = Dataset::open_with_storage(storage).unwrap();
    assert_eq!(ds.variables()["quality"].role, Role::Coordinate);
    assert!(ds.variables()["total"].dims.is_empty());
    assert!(!ds.variables().contains_key("extension"));
    assert!(!ds.variables().contains_key("hidden"));
    assert!(matches!(
        ds.variables()["unsupported"].logical_support,
        Support::Unsupported { .. }
    ));
    let trace = String::from_utf8(log.lock().unwrap().clone()).unwrap();
    assert!(!trace.contains("/c/"), "{trace}");
    assert!(!trace.contains("hidden/zarr.json"), "{trace}");
}
#[test]
fn unsupported_version_is_distinct_from_bad_identity() {
    let (storage, _) = catalog_fixture("contig", "contig_length");
    let mut value = root("contig", "contig_length");
    value["attributes"]["pbz:version"] = json!("future");
    put(&storage, "zarr.json", value);
    assert!(matches!(
        Dataset::open_with_storage(storage.clone()),
        Err(PbzError::UnsupportedFormat(_))
    ));
    let mut value = root("contig", "contig_length");
    value["attributes"]["zarr_conventions"] = json!([]);
    put(&storage, "zarr.json", value);
    assert!(matches!(
        Dataset::open_with_storage(storage),
        Err(PbzError::Metadata(_))
    ));
}
