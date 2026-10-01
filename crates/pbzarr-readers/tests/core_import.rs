// One range per axis; the single-range cases are rank-one reads.
#![allow(clippy::single_range_in_vec_init)]
mod bam_common;
mod common;

use pbzarr::TrackConfig;
use pbzarr::dataset::{Dataset, Reference};
use pbzarr::import::Import;
use pbzarr::io::{Dtype, ValueReader};
use pbzarr_readers::{BamReader, BigWigReader, DepthFilter, ImportMode};
use tempfile::TempDir;

#[test]
fn bigwig_imports_to_root_with_singleton_sample_and_gap_fill() {
    let dir = TempDir::new().unwrap();
    let source = common::write_bigwig(
        dir.path(),
        "signal",
        &[("chr1", 16)],
        &[("chr1", 0, 4, 2.5), ("chr1", 8, 16, 7.0)],
        false,
    );
    let reader = BigWigReader::open(&source).unwrap();
    let reference = Reference::new(reader.contigs().contigs().to_vec()).unwrap();
    let path = dir.path().join("signal.pbz");
    let mut ds = Dataset::create(&path, reference).unwrap();
    Import::from_readers(vec![reader])
        .unwrap()
        .into_dataset(
            &mut ds,
            vec![(
                "signal".into(),
                TrackConfig::new(Dtype::F32)
                    .chunk_size(4)
                    .columns(vec!["S1".into()])
                    .column_dim("sample"),
            )],
        )
        .readers_as_columns()
        .run()
        .unwrap();
    let ds = Dataset::open(path).unwrap();
    let values = ds.read_raw::<f32>("signal", &[0..16, 0..1]).unwrap();
    assert_eq!(values.shape(), [16, 1]);
    let values = values.as_slice().unwrap();
    assert_eq!(&values[..4], &[2.5; 4]);
    assert!(values[4..8].iter().all(|v| v.is_nan()));
    assert_eq!(&values[8..], &[7.; 8]);
}
#[test]
fn bam_depth_imports_to_root_with_uncovered_contig_fill() {
    let fixture = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let reader = BamReader::open(
        &fixture.bam,
        None,
        ImportMode::Depth,
        DepthFilter::default(),
    )
    .unwrap();
    let reference = Reference::new(reader.contigs().contigs().to_vec()).unwrap();
    let path = dir.path().join("depth.pbz");
    let mut ds = Dataset::create(&path, reference).unwrap();
    Import::from_readers(vec![reader])
        .unwrap()
        .into_dataset(
            &mut ds,
            vec![("depth".into(), TrackConfig::new(Dtype::I32).chunk_size(100))],
        )
        .run()
        .unwrap();
    let ds = Dataset::open(path).unwrap();
    assert_eq!(
        ds.read_raw::<i32>("depth", &[0..100])
            .unwrap()
            .as_slice()
            .unwrap(),
        &bam_common::expected_default()
    );
    assert!(
        ds.read_raw::<i32>("depth", &[100..160])
            .unwrap()
            .iter()
            .all(|&n| n == 0)
    );
}

#[test]
fn bam_composition_shares_sample_coordinates_across_fields() {
    let fixture = bam_common::fixture();
    let dir = TempDir::new().unwrap();
    let reader = BamReader::open(
        &fixture.bam,
        None,
        ImportMode::Composition,
        DepthFilter {
            min_bq: 20,
            ..DepthFilter::default()
        },
    )
    .unwrap();
    let reference = Reference::new(reader.contigs().contigs().to_vec()).unwrap();
    let configs = reader
        .output_schema()
        .fields
        .iter()
        .map(|field| {
            (
                field.name.clone().unwrap(),
                TrackConfig::new(field.dtype)
                    .chunk_size(20)
                    .columns(vec!["S1".into(), "S2".into()])
                    .column_dim("sample"),
            )
        })
        .collect();
    let second = reader.fork().unwrap();
    let mut ds = Dataset::create(dir.path().join("composition.pbz"), reference).unwrap();
    Import::from_readers(vec![reader, second])
        .unwrap()
        .into_dataset(&mut ds, configs)
        .fields_as_tracks()
        .readers_as_columns()
        .run()
        .unwrap();
    assert_eq!(
        ds.validate().unwrap().status,
        pbzarr::dataset::ValidationStatus::Valid
    );
    assert_eq!(ds.variables()["ins"].dims, ["position", "sample"]);
    assert_eq!(
        ds.read_raw::<String>("sample", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &["S1", "S2"]
    );
    let ins = ds.read_raw::<i32>("ins", &[0..160, 0..2]).unwrap();
    for ((position, _), &value) in ins
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap()
        .indexed_iter()
    {
        assert_eq!(value, i32::from(position == 54));
    }
    assert!(
        ds.read_raw::<i32>("del", &[50..55, 0..2])
            .unwrap()
            .iter()
            .all(|&n| n == 1)
    );
    assert_eq!(
        ds.read_raw::<i32>("n", &[15..16, 0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, 1]
    );
}
