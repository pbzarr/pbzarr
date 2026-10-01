// Each range selects one axis; a single range intentionally describes rank one.
#![allow(clippy::single_range_in_vec_init)]
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

use pbzarr::dataset::{GenomicAxis, ValidationStatus};
use zarrs::array::{Array, ArraySubset, Element};
fn values<T: Element>(storage: &ReadableWritableListableStorage, name: &str, data: &[T]) {
    let array = Array::open(storage.clone(), &format!("/{name}")).unwrap();
    array
        .store_array_subset(&ArraySubset::new_with_shape(array.shape().to_vec()), data)
        .unwrap();
}
fn fixture() -> (ReadableWritableListableStorage, Trace) {
    let (s, log) = catalog_fixture("contig", "contig_length");
    values(&s, "contig", &["chr1".to_owned(), "chr2".to_owned()]);
    values(&s, "contig_length", &[5_i64, 7]);
    values(
        &s,
        "sample",
        &["s1".to_owned(), "s2".to_owned(), "s3".to_owned()],
    );
    log.lock().unwrap().clear();
    (s, log)
}
fn has_rule(s: ReadableWritableListableStorage, rule: &str) {
    let report = Dataset::open_with_storage(s).unwrap().validate().unwrap();
    assert_eq!(report.status, ValidationStatus::Invalid, "{report:?}");
    assert!(report.issues.iter().any(|i| i.rule == rule), "{report:?}");
}
#[test]
fn validates_reference_and_coordinates_without_reading_measurements() {
    let (s, log) = fixture();
    let ds = Dataset::open_with_storage(s).unwrap();
    assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    assert_eq!(
        ds.reference()
            .unwrap()
            .extent(&GenomicAxis::Position)
            .unwrap(),
        12
    );
    let trace = String::from_utf8(log.lock().unwrap().clone()).unwrap();
    assert!(!trace.contains("depth/c"), "{trace}");
}
#[test]
fn validates_metadata_rules() {
    let cases: Vec<(&str, &str, Value)> = vec![
        ("R2", "zarr.json", {
            let mut m = root("contig", "contig");
            m["attributes"]["coordinates"] = json!("contig_length");
            m
        }),
        ("R3", "zarr.json", {
            let mut m = root("contig", "contig_length");
            m["attributes"]
                .as_object_mut()
                .unwrap()
                .remove("coordinates");
            m
        }),
        (
            "R5",
            "contig_length/zarr.json",
            metadata(&[2], &["contig"], "float64"),
        ),
        (
            "D1",
            "depth/zarr.json",
            metadata(&[12, 3], &["position", ""], "uint16"),
        ),
        (
            "D1",
            "depth/zarr.json",
            metadata(&[12, 3], &["position", "position"], "uint16"),
        ),
        (
            "D2",
            "depth/zarr.json",
            metadata(&[12, 2], &["position", "sample"], "uint16"),
        ),
        (
            "D3",
            "depth/zarr.json",
            metadata(&[12, 3], &["position", "unlabeled"], "uint16"),
        ),
        (
            "G1",
            "depth/zarr.json",
            metadata(&[4, 3], &["bin_04", "sample"], "uint16"),
        ),
        (
            "G2",
            "depth/zarr.json",
            metadata(&[11, 3], &["position", "sample"], "uint16"),
        ),
        (
            "G2",
            "depth/zarr.json",
            metadata(&[3, 3], &["bin_4", "sample"], "uint16"),
        ),
        (
            "G3",
            "depth/zarr.json",
            metadata(&[3, 12], &["sample", "position"], "uint16"),
        ),
        (
            "G3",
            "depth/zarr.json",
            metadata(&[12, 4], &["position", "bin_4"], "uint16"),
        ),
        (
            "G4",
            "position/zarr.json",
            metadata(&[12], &["position"], "uint64"),
        ),
    ];
    for (rule, key, value) in cases {
        let (s, _) = fixture();
        put(&s, key, value);
        has_rule(s, rule);
    }
}
#[test]
fn rejects_bad_reference_values_and_missing_dimension_labels() {
    let (s, _) = fixture();
    values(&s, "contig", &["chr1".to_owned(), "chr1".to_owned()]);
    has_rule(s, "R4");
    let (s, _) = fixture();
    values(&s, "contig_length", &[5_i64, -1]);
    has_rule(s, "R6");
    let (s, _) = fixture();
    values(&s, "contig_length", &[i64::MAX, 1_i64]);
    has_rule(s, "R6");
    let (s, _) = fixture();
    values(
        &s,
        "sample",
        &["s1".to_owned(), "s1".to_owned(), "s3".to_owned()],
    );
    has_rule(s, "D3");
    let (s, _) = fixture();
    let mut m = metadata(&[3], &["sample"], "int64");
    m["attributes"]["_FillValue"] = json!(-1);
    put(&s, "sample/zarr.json", m);
    values(&s, "sample", &[1_i64, 2, -1]);
    has_rule(s, "D3");
}
#[test]
fn numeric_labels_stay_exact_and_auxiliary_labels_may_repeat() {
    let (s, _) = fixture();
    put(
        &s,
        "sample/zarr.json",
        metadata(&[3], &["sample"], "uint64"),
    );
    values(
        &s,
        "sample",
        &[1_u64 << 53, (1_u64 << 53) + 1, (1_u64 << 53) + 2],
    );
    let mut m = root("contig", "contig_length");
    m["attributes"]["coordinates"] = json!("contig_length population");
    put(&s, "zarr.json", m);
    put(
        &s,
        "population/zarr.json",
        metadata(&[3], &["sample"], "string"),
    );
    values(
        &s,
        "population",
        &["p1".to_owned(), "p1".to_owned(), "p2".to_owned()],
    );
    put(&s, "one/zarr.json", metadata(&[12], &["bin_1"], "uint16"));
    put(
        &s,
        "huge/zarr.json",
        metadata(
            &[2],
            &["bin_99999999999999999999999999999999999999999"],
            "uint16",
        ),
    );
    assert_eq!(
        Dataset::open_with_storage(s)
            .unwrap()
            .validate()
            .unwrap()
            .status,
        ValidationStatus::Valid
    );
}
#[test]
fn float_zero_labels_collide_and_nan_is_missing() {
    for labels in [[0.0_f64, -0.0, 1.0], [0.0, 1.0, f64::NAN]] {
        let (s, _) = fixture();
        put(
            &s,
            "sample/zarr.json",
            metadata(&[3], &["sample"], "float64"),
        );
        values(&s, "sample", &labels);
        has_rule(s, "D3");
    }
}
#[test]
fn unavailable_required_encoding_is_incomplete_but_does_not_hide_invalidity() {
    let (s, log) = fixture();
    let mut m = metadata(&[3], &["sample"], "string");
    m["codecs"] = json!([{"name":"unknown"}]);
    put(&s, "sample/zarr.json", m);
    let report = Dataset::open_with_storage(s.clone())
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(report.status, ValidationStatus::Incomplete, "{report:?}");
    put(
        &s,
        "depth/zarr.json",
        metadata(&[11, 3], &["position", "sample"], "uint16"),
    );
    has_rule(s, "G2");
    assert!(
        !String::from_utf8(log.lock().unwrap().clone())
            .unwrap()
            .contains("depth/c")
    );
}
#[test]
fn reference_order_is_preserved_and_unused_missing_sentinels_are_valid() {
    let (s, _) = fixture();
    let mut m = metadata(&[2], &["contig"], "int64");
    m["attributes"]["_FillValue"] = json!(-1);
    put(&s, "contig_length/zarr.json", m);
    let ds = Dataset::open_with_storage(s.clone()).unwrap();
    assert_eq!(ds.reference().unwrap().contigs()[0].name, "chr1");
    values(&s, "contig", &["chr2".to_owned(), "chr1".to_owned()]);
    let other = Dataset::open_with_storage(s).unwrap();
    assert_eq!(other.reference().unwrap().contigs()[0].name, "chr2");
    assert_eq!(other.validate().unwrap().status, ValidationStatus::Valid);
}

#[test]
fn convention_identity_uses_uuid_and_rejects_duplicate_markers() {
    let (s, _) = fixture();
    let mut m = root("contig", "contig_length");
    m["attributes"]["zarr_conventions"][0]
        .as_object_mut()
        .unwrap()
        .remove("name");
    put(&s, "zarr.json", m.clone());
    assert_eq!(
        Dataset::open_with_storage(s.clone())
            .unwrap()
            .validate()
            .unwrap()
            .status,
        ValidationStatus::Valid
    );
    let marker = m["attributes"]["zarr_conventions"][0].clone();
    m["attributes"]["zarr_conventions"]
        .as_array_mut()
        .unwrap()
        .push(marker);
    put(&s, "zarr.json", m);
    assert!(Dataset::open_with_storage(s).is_err());
}
#[test]
fn integer_missing_sentinel_written_as_json_float_is_still_missing() {
    let (s, _) = fixture();
    let mut m = metadata(&[2], &["contig"], "int64");
    m["attributes"]["_FillValue"] = json!(5.0);
    put(&s, "contig_length/zarr.json", m);
    has_rule(s, "R6");
}

fn read_fixture() -> (ReadableWritableListableStorage, Trace) {
    let (s, log) = fixture();
    let mut m = metadata(&[12, 3], &["position", "sample"], "uint16");
    m["chunk_grid"]["configuration"]["chunk_shape"] = json!([4, 3]);
    put(&s, "depth/zarr.json", m);
    values(&s, "depth", &(0..36_u16).collect::<Vec<_>>());
    put(
        &s,
        "threshold/zarr.json",
        metadata(&[2], &["threshold"], "int64"),
    );
    values(&s, "threshold", &[5_i64, 10]);
    let mut m = metadata(&[12, 3, 2], &["position", "sample", "threshold"], "int8");
    m["attributes"]["dtype"] = json!("bool");
    put(&s, "callable/zarr.json", m);
    values(
        &s,
        "callable",
        &(0..72)
            .map(|n| i8::from((n / 2) >= if n % 2 == 0 { 5 } else { 10 }))
            .collect::<Vec<_>>(),
    );
    put(
        &s,
        "coverage/zarr.json",
        metadata(&[4, 3], &["bin_4", "sample"], "float32"),
    );
    values(
        &s,
        "coverage",
        &(0..12).map(|n| n as f32).collect::<Vec<_>>(),
    );
    put(&s, "mean/zarr.json", metadata(&[3], &["sample"], "float64"));
    values(&s, "mean", &[1.5_f64, 2.5, 3.5]);
    put(&s, "large_unsigned/zarr.json", metadata(&[], &[], "uint64"));
    values(&s, "large_unsigned", &[(1_u64 << 53) + 1]);
    put(&s, "large_signed/zarr.json", metadata(&[], &[], "int64"));
    values(&s, "large_signed", &[(1_i64 << 53) + 1]);
    let mut m = metadata(&[12], &["position"], "int16");
    m["attributes"]["_FillValue"] = json!(-1);
    put(&s, "missing_count/zarr.json", m);
    values(
        &s,
        "missing_count",
        &[0_i16, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
    );
    log.lock().unwrap().clear();
    (s, log)
}
#[test]
fn reads_selected_values_without_losing_rank_or_integer_precision() {
    let (s, _) = read_fixture();
    let ds = Dataset::open_with_storage(s).unwrap();
    let a = ds.read_raw::<u16>("depth", &[2..4, 1..3]).unwrap();
    assert_eq!(a.shape(), [2, 2]);
    assert_eq!(a.iter().copied().collect::<Vec<_>>(), [7, 8, 10, 11]);
    let a = ds.read_bool("callable", &[2..4, 1..3, 0..2]).unwrap();
    assert_eq!(a.shape(), [2, 2, 2]);
    assert_eq!(
        a.iter().copied().collect::<Vec<_>>(),
        [true, false, true, false, true, true, true, true]
    );
    assert_eq!(
        ds.read_raw::<f32>("coverage", &[2..4, 0..1])
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [6.0, 9.0]
    );
    assert_eq!(
        ds.read_raw::<f64>("mean", &[1..3])
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [2.5, 3.5]
    );
    let a = ds.read_raw::<u64>("large_unsigned", &[]).unwrap();
    assert!(a.shape().is_empty());
    assert_eq!(a[ndarray::IxDyn(&[])], (1_u64 << 53) + 1);
    assert_eq!(
        ds.read_raw::<i64>("large_signed", &[]).unwrap()[ndarray::IxDyn(&[])],
        (1_i64 << 53) + 1
    );
    assert_eq!(
        ds.read_raw::<i16>("missing_count", &[0..3])
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0, 1, -1]
    );
}
#[test]
fn checks_bounds_and_dtype_before_payload_and_reads_only_selected_chunks() {
    let (s, log) = read_fixture();
    let ds = Dataset::open_with_storage(s).unwrap();
    log.lock().unwrap().clear();
    assert!(ds.read_raw::<u16>("depth", &[0..13, 0..1]).is_err());
    assert!(ds.read_raw::<u16>("depth", &[0..1]).is_err());
    assert!(ds.read_raw::<i16>("depth", &[0..1, 0..1]).is_err());
    assert!(
        !String::from_utf8(log.lock().unwrap().clone())
            .unwrap()
            .contains("/c")
    );
    let a = ds.read_raw::<u16>("depth", &[2..2, 0..3]).unwrap();
    assert_eq!(a.shape(), [0, 3]);
    assert!(
        !String::from_utf8(log.lock().unwrap().clone())
            .unwrap()
            .contains("depth/c")
    );
    log.lock().unwrap().clear();
    ds.read_raw::<u16>("depth", &[1..2, 1..2]).unwrap();
    let trace = String::from_utf8(log.lock().unwrap().clone()).unwrap();
    assert!(trace.contains("depth/c/0/0"), "{trace}");
    assert!(
        !trace.contains("depth/c/1/0") && !trace.contains("depth/c/2/0"),
        "{trace}"
    );
}
#[test]
fn masked_bool_is_explicitly_unsupported_but_raw_values_and_neighbors_are_readable() {
    let (s, _) = read_fixture();
    let mut m = metadata(&[12], &["position"], "int8");
    m["attributes"] = json!({"dtype":"bool","_FillValue":0});
    put(&s, "masked/zarr.json", m);
    values(&s, "masked", &[0_i8, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
    let mut unknown = metadata(&[12], &["position"], "uint16");
    unknown["codecs"] = json!([{"name":"unknown"}]);
    put(&s, "unsupported/zarr.json", unknown);
    let ds = Dataset::open_with_storage(s).unwrap();
    assert!(matches!(
        ds.read_bool("masked", &[0..2]),
        Err(PbzError::UnsupportedEncoding(_))
    ));
    assert_eq!(
        ds.read_raw::<i8>("masked", &[0..2])
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(ds.read_raw::<u16>("depth", &[0..1, 0..1]).is_ok());
}
#[test]
fn validates_read_dependencies_even_without_full_validation() {
    let (s, _) = read_fixture();
    values(
        &s,
        "sample",
        &["s1".to_owned(), "s1".to_owned(), "s3".to_owned()],
    );
    assert!(matches!(
        Dataset::open_with_storage(s)
            .unwrap()
            .read_raw::<u16>("depth", &[0..1, 0..1]),
        Err(PbzError::Metadata(_))
    ));
}
#[test]
fn supports_physical_boolean_values() {
    let (s, _) = fixture();
    let mut m = metadata(&[12], &["position"], "bool");
    m["fill_value"] = json!(false);
    put(&s, "boolean/zarr.json", m);
    values(
        &s,
        "boolean",
        &[
            true, false, true, false, true, false, true, false, true, false, true, false,
        ],
    );
    assert_eq!(
        Dataset::open_with_storage(s)
            .unwrap()
            .read_bool("boolean", &[0..3])
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [true, false, true]
    );
}
