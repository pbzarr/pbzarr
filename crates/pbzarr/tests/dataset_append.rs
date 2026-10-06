//! Column append onto a format-0.5 root Dataset.
#![allow(clippy::single_range_in_vec_init)]
use std::sync::Arc;

use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::{Axes, Config, Import, PipelineOptions, Target};
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, Genome, TrackConfig};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage,
};

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

fn genome() -> Genome {
    Genome::new(reference().contigs().to_vec()).unwrap()
}

/// Serves one constant value everywhere.
#[derive(Clone)]
struct ConstReader {
    genome: Genome,
    schema: OutputSchema,
    value: u32,
}

impl ConstReader {
    fn new(value: u32) -> Self {
        Self {
            genome: genome(),
            schema: OutputSchema::single("value", Dtype::U32),
            value,
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
        outputs[0].as_u32_mut()?.fill(self.value);
        Ok(())
    }
}

fn cohort(labels: &[&str], column_chunk: usize) -> TrackConfig {
    TrackConfig::new(Dtype::U32)
        .columns(labels.iter().map(|l| (*l).to_owned()).collect())
        .column_dim("sample")
        .column_chunk_size(column_chunk)
        .chunk_size(4)
}

fn import(dataset: &mut Dataset, track: &str, labels: &[&str], column_chunk: usize, base: u32) {
    let readers: Vec<ConstReader> = (0..labels.len())
        .map(|i| ConstReader::new(base + i as u32))
        .collect();
    Import::from_readers(readers)
        .unwrap()
        .into_dataset(dataset, vec![(track.into(), cohort(labels, column_chunk))])
        .readers_as_columns()
        .expect_column_labels(labels.iter().map(|l| (*l).to_owned()).collect())
        .run()
        .unwrap();
}

fn append(
    dataset: &mut Dataset,
    track: &str,
    labels: &[&str],
    base: u32,
) -> pbzarr::Result<pbzarr::import::Report> {
    let readers: Vec<ConstReader> = (0..labels.len())
        .map(|i| ConstReader::new(base + i as u32))
        .collect();
    let owned: Vec<String> = labels.iter().map(|l| (*l).to_owned()).collect();
    Target::Append(dataset).append(
        vec![(track.into(), genome(), cohort(labels, 2))],
        readers,
        Axes::default().readers_as_columns(owned),
        PipelineOptions::default(),
        &Config::default(),
    )
}

fn store() -> (Dataset, ReadableWritableListableStorage) {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let dataset = Dataset::create_with_storage(storage.clone(), reference()).unwrap();
    (dataset, storage)
}

/// Chunk payloads, so a rewritten chunk file shows up as a changed value.
fn chunks(storage: &ReadableWritableListableStorage) -> Vec<(String, Vec<u8>)> {
    storage
        .list()
        .unwrap()
        .iter()
        .filter(|key| !key.as_str().ends_with("zarr.json"))
        .map(|key| {
            (
                key.as_str().to_owned(),
                storage.get(key).unwrap().unwrap().to_vec(),
            )
        })
        .collect()
}

#[test]
fn append_adds_columns_and_leaves_the_existing_chunks_alone() {
    let (mut ds, storage) = store();
    import(&mut ds, "depth", &["s0", "s1"], 2, 1);
    let before = chunks(&storage);
    let stored: Vec<_> = ["depth", "sample"]
        .map(|name| (name, ds.variables()[name].metadata.clone()))
        .into();

    append(&mut ds, "depth", &["s2", "s3"], 3).unwrap();

    // Growing publishes a new shape and nothing else: a codec respelled on
    // the way out would make the store unreadable to other Zarr libraries.
    for (name, before) in &stored {
        let mut after = ds.variables()[*name].metadata.clone();
        after["shape"] = before["shape"].clone();
        assert_eq!(&after, before, "{name} metadata changed beyond its shape");
    }

    assert_eq!(ds.variables()["depth"].shape, [12, 4]);
    assert_eq!(
        ds.read_raw::<String>("sample", &[0..4])
            .unwrap()
            .as_slice()
            .unwrap(),
        &["s0", "s1", "s2", "s3"]
    );
    let values = ds.read_raw::<u32>("depth", &[0..12, 0..4]).unwrap();
    assert_eq!(
        values,
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[12, 4]), |i| 1 + i[1] as u32)
    );

    // The coordinate rewrites its own chunk; no measurement chunk does.
    for (key, payload) in &before {
        if key.starts_with("sample/") {
            continue;
        }
        assert_eq!(
            storage.get(&key.as_str().try_into().unwrap()).unwrap(),
            Some(payload.clone().into()),
            "{key} was rewritten"
        );
    }
    let keys: Vec<String> = chunks(&storage).into_iter().map(|(key, _)| key).collect();
    assert!(keys.iter().any(|key| key == "depth/c/0/1"), "{keys:?}");
}

#[test]
fn append_grows_every_variable_over_the_dimension() {
    let (mut ds, _) = store();
    import(&mut ds, "depth", &["s0", "s1"], 2, 1);
    import(&mut ds, "other", &["s0", "s1"], 2, 7);

    append(&mut ds, "depth", &["s2", "s3"], 3).unwrap();

    assert_eq!(ds.variables()["other"].shape, [12, 4]);
    let other = ds.read_raw::<u32>("other", &[0..12, 0..4]).unwrap();
    for row in other.rows() {
        assert_eq!(row.to_vec(), [7, 8, 0, 0]);
    }
}

#[test]
fn append_refuses_a_width_that_is_not_a_whole_number_of_column_chunks() {
    let (mut ds, _) = store();
    import(&mut ds, "depth", &["s0", "s1", "s2"], 2, 1);

    let error = match append(&mut ds, "depth", &["s3", "s4"], 4) {
        Err(error) => error.to_string(),
        Ok(_) => panic!("append accepted an unaligned width"),
    };

    assert!(error.contains("multiple of 2"), "{error}");
    assert_eq!(ds.variables()["depth"].shape, [12, 3]);
}
