//! End-to-end: synthesize multi-column bgzipped+tabix BEDs and import them via
//! the schema generation (`plan_bed_schema`/`execute_bed_schema_plan`) and
//! `from_bed_matrix`, then read the tracks back.
// One range per axis; the single-range cases are rank-one reads.
#![allow(clippy::single_range_in_vec_init)]

mod common;

use std::path::{Path, PathBuf};

use ndarray::Ix2;
use pbzarr::Contig;
use pbzarr::RegionQuery;
use pbzarr::import::Config;
use pbzarr::io::Dtype;
use pbzarr::{Dataset, Fill, Reference};
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, Source, execute_bed_schema_plan, from_bed_matrix,
    plan_bed_schema,
};
use tempfile::TempDir;

use common::{write_bed_bgzip_tabix, write_headerless_bed_bgzip_tabix};

fn genome(cs: &[(&str, u64)]) -> Reference {
    Reference::new(
        cs.iter()
            .map(|(n, l)| Contig {
                name: (*n).to_owned(),
                length: *l,
            })
            .collect(),
    )
    .unwrap()
}

/// An empty store over one `chr1` contig of `length` positions.
fn create(dir: &Path, name: &str, length: u64) -> (PathBuf, Dataset) {
    let path = dir.join(name);
    let reference = Reference::new(genome(&[("chr1", length)]).contigs().to_vec()).unwrap();
    let dataset = Dataset::create(&path, reference).unwrap();
    (path, dataset)
}

fn track_names(store: &Path) -> Vec<String> {
    Dataset::open(store)
        .unwrap()
        .variables()
        .values()
        .filter(|v| v.is_track())
        .map(|v| v.name.clone())
        .collect()
}

fn labels(dataset: &Dataset, dim: &str) -> Vec<String> {
    let n = dataset.variables()[dim].shape[0];
    dataset
        .read_raw::<String>(dim, &[0..n])
        .unwrap()
        .into_raw_vec_and_offset()
        .0
}

fn chr1() -> RegionQuery {
    "chr1".parse().unwrap()
}

#[test]
fn schema_plan_rejects_an_ambiguous_named_bed_column() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "duplicate-header",
        &["chrom", "start", "end", "value", "value"],
        &[("chr1", 0, 8, vec!["4", "9"])],
    );

    let result = plan_bed_schema(
        &[Source::new(bed)],
        &BedSchema(vec![BedColumnSpec::named("value", Dtype::U16)]),
        &Config::default(),
    );
    let error = match result {
        Ok(_) => panic!("a named selector must not pick the first duplicate header"),
        Err(error) => error.to_string(),
    };

    assert!(error.contains("BED column \"value\""), "{error}");
    assert!(error.contains("ambiguous"), "{error}");
    assert!(error.contains("2 header columns"), "{error}");
}

#[test]
fn indexed_duplicate_header_labels_are_safe_for_per_column_tracks() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "duplicate-header-indexed",
        &["chrom", "start", "end", "value", "value"],
        &[("chr1", 0, 8, vec!["4", "9"])],
    );
    let sources = [Source::new(bed)];
    let schema = BedSchema(vec![
        BedColumnSpec::indexed(3, Dtype::U16, "first"),
        BedColumnSpec::indexed(4, Dtype::U16, "second"),
    ]);

    let plan = plan_bed_schema(&sources, &schema, &Config::default())
        .expect("duplicate labels are not persisted by PerColumn layout");
    let output = dir.path().join("duplicate-header-indexed.pbz");
    execute_bed_schema_plan(
        output.as_path(),
        plan,
        genome(&[("chr1", 8)]),
        Config::default(),
    )
    .unwrap();

    let dataset = Dataset::open(&output).unwrap();
    for (track_name, expected) in [("first", 4u16), ("second", 9u16)] {
        let values = dataset.read_region::<u16>(track_name, &chr1()).unwrap();
        assert!(values.iter().all(|value| *value == expected));
    }
}

#[test]
fn indexed_schema_without_a_target_names_the_selector_one_based() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "missing-target",
        &["chrom", "start", "end", "value"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let mut column = BedColumnSpec::indexed(3, Dtype::U16, "unused");
    column.track_name = None;

    let error = plan_bed_schema(
        &[Source::new(bed)],
        &BedSchema(vec![column]),
        &Config::default(),
    )
    .err()
    .expect("an indexed selector has no header name to use as a target")
    .to_string();

    assert!(
        error.contains("BED column 4 (one-based) needs an explicit target track name"),
        "{error}"
    );
    assert!(!error.contains("BED column 3 needs"), "{error}");
}

#[test]
fn grouped_indexed_columns_reject_duplicate_persisted_labels() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "duplicate-grouped-labels",
        &["chrom", "start", "end", "value", "value"],
        &[("chr1", 0, 8, vec!["4", "9"])],
    );
    let schema = BedSchema(vec![
        BedColumnSpec::indexed(3, Dtype::U16, "stats"),
        BedColumnSpec::indexed(4, Dtype::U16, "stats"),
    ]);

    let result = plan_bed_schema(
        &[Source::new(bed)],
        &schema,
        &Config {
            column_dim: Some("metric".into()),
            ..Config::default()
        },
    );
    let error = match result {
        Ok(_) => panic!("grouped labels are persisted and must be unique"),
        Err(error) => error.to_string(),
    };

    assert!(error.contains("grouped track \"stats\""), "{error}");
    assert!(
        error.contains("duplicate BED column label \"value\""),
        "{error}"
    );
}

#[test]
fn grouped_schema_requires_exactly_matching_descriptions() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "grouped-description-mismatch",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["4", "9"])],
    );

    for (first_description, second_description) in
        [(Some("shared"), None), (Some("first"), Some("second"))]
    {
        let mut cov = BedColumnSpec::named("cov", Dtype::U16);
        cov.track_name = Some("stats".into());
        cov.description = first_description.map(str::to_owned);
        let mut mapq = BedColumnSpec::named("mapq", Dtype::U16);
        mapq.track_name = Some("stats".into());
        mapq.description = second_description.map(str::to_owned);

        let result = plan_bed_schema(
            &[Source::new(&bed)],
            &BedSchema(vec![cov, mapq]),
            &Config {
                column_dim: Some("metric".into()),
                ..Config::default()
            },
        );
        let error = match result {
            Ok(_) => panic!("grouped descriptions must match exactly"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("track \"stats\""), "{error}");
        assert!(error.contains("descriptions"), "{error}");
        assert!(error.contains("match exactly"), "{error}");
    }
}

#[test]
fn grouped_schema_one_source_preserves_bed_column_order() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "mapq"],
        &[
            ("chr1", 0, 4, vec!["4", "60"]),
            ("chr1", 4, 8, vec!["9", "30"]),
        ],
    );
    let sources = [Source::new(bed)];
    let mut mapq = BedColumnSpec::named("mapq", Dtype::U16);
    mapq.track_name = Some("stats".into());
    mapq.description = Some("Quality statistics".into());
    let mut cov = BedColumnSpec::named("cov", Dtype::U16);
    cov.track_name = Some("stats".into());
    cov.description = Some("Quality statistics".into());
    let schema = BedSchema(vec![mapq, cov]);
    let config = Config {
        column_dim: Some("BED column".into()),
        ..Config::default()
    };

    let plan = plan_bed_schema(&sources, &schema, &config).unwrap();
    let output = dir.path().join("grouped.pbz");
    execute_bed_schema_plan(output.as_path(), plan, genome(&[("chr1", 8)]), config).unwrap();

    let dataset = Dataset::open(&output).unwrap();
    assert_eq!(
        dataset.variables()["stats"].dims,
        ["position", "BED column"]
    );
    assert_eq!(labels(&dataset, "BED column"), ["mapq", "cov"]);
    let values = dataset
        .read_region::<u16>("stats", &chr1())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (index, row) in values.rows().into_iter().enumerate() {
        assert_eq!(
            row.to_vec(),
            if index < 4 { vec![60, 4] } else { vec![30, 9] }
        );
    }
}

#[test]
fn unique_schema_two_sources_uses_labeled_source_axis_per_track() {
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["4", "60"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["9", "30"])],
    );
    let sources = [
        Source::labeled(first, "tumor"),
        Source::labeled(second, "normal"),
    ];
    let schema = BedSchema(vec![
        BedColumnSpec::named("mapq", Dtype::U16),
        BedColumnSpec::named("cov", Dtype::U16),
    ]);
    let config = Config {
        column_dim: Some("sample".into()),
        ..Config::default()
    };
    let plan = plan_bed_schema(&sources, &schema, &config).unwrap();
    let output = dir.path().join("unique.pbz");
    execute_bed_schema_plan(output.as_path(), plan, genome(&[("chr1", 8)]), config).unwrap();

    let dataset = Dataset::open(&output).unwrap();
    assert_eq!(labels(&dataset, "sample"), ["tumor", "normal"]);
    for (track_name, expected) in [("mapq", vec![60, 30]), ("cov", vec![4, 9])] {
        assert_eq!(dataset.variables()[track_name].dims, ["position", "sample"]);
        let values = dataset
            .read_region::<u16>(track_name, &chr1())
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        assert!(
            values
                .rows()
                .into_iter()
                .all(|row| row.to_vec() == expected)
        );
    }
}

#[test]
fn grouped_schema_two_sources_reports_rank_three_before_track_creation() {
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["4", "60"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, vec!["9", "30"])],
    );
    let sources = [Source::new(first), Source::new(second)];
    let mut cov = BedColumnSpec::named("cov", Dtype::U16);
    cov.track_name = Some("stats".into());
    let mut mapq = BedColumnSpec::named("mapq", Dtype::U16);
    mapq.track_name = Some("stats".into());
    let schema = BedSchema(vec![cov, mapq]);
    let config = Config {
        column_dim: Some("metric".into()),
        ..Config::default()
    };

    let error = plan_bed_schema(&sources, &schema, &config)
        .err()
        .expect("grouped multi-source schema must fail")
        .to_string();
    assert!(
        error.contains("2 BED columns map to track \"stats\""),
        "{error}"
    );
    assert!(error.contains("BED-column axis \"metric\""), "{error}");
    assert!(error.contains("2 sources create a source axis"), "{error}");
    assert!(
        error.contains("rank 3 (position, source, metric)"),
        "{error}"
    );
    assert!(
        error.contains("PBZ supports only one non-position axis"),
        "{error}"
    );
    assert!(
        error.contains("Use one input or give every BED column a unique track name"),
        "{error}"
    );
}

#[test]
fn unique_schema_two_sources_requires_column_dimension() {
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["9"])],
    );
    let error = plan_bed_schema(
        &[Source::new(first), Source::new(second)],
        &BedSchema(vec![BedColumnSpec::named("cov", Dtype::U16)]),
        &Config::default(),
    )
    .err()
    .expect("multi-source per-BED-column layout must require an axis name")
    .to_string();
    assert!(error.contains("BED column"), "{error}");
    assert!(error.contains("rank 2 (position, source)"), "{error}");
    assert!(error.contains("column dimension name"), "{error}");
}

#[test]
fn partially_grouped_schema_is_rejected() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "one",
        &["chrom", "start", "end", "cov", "mapq", "score"],
        &[("chr1", 0, 8, vec!["4", "60", "1.5"])],
    );
    let mut cov = BedColumnSpec::named("cov", Dtype::U16);
    cov.track_name = Some("stats".into());
    let mut mapq = BedColumnSpec::named("mapq", Dtype::U16);
    mapq.track_name = Some("stats".into());
    let schema = BedSchema(vec![cov, mapq, BedColumnSpec::named("score", Dtype::F32)]);
    let error = plan_bed_schema(
        &[Source::new(bed)],
        &schema,
        &Config {
            column_dim: Some("metric".into()),
            ..Config::default()
        },
    )
    .err()
    .expect("partial grouping must fail")
    .to_string();
    assert!(error.contains("partially grouped BED columns"), "{error}");
    assert!(error.contains("all BED columns to one track"), "{error}");
    assert!(error.contains("unique track name"), "{error}");
}

#[test]
fn schema_plan_errors_are_shape_aware_and_preflighted() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "one",
        &["chrom", "start", "end", "cov", "mapq", "score"],
        &[("chr1", 0, 8, vec!["4", "60", "1.5"])],
    );
    let sources = [Source::new(&bed)];

    let mut cov = BedColumnSpec::named("cov", Dtype::U16);
    cov.track_name = Some("stats".into());
    let mut mapq = BedColumnSpec::named("mapq", Dtype::U16);
    mapq.track_name = Some("stats".into());
    let grouped = BedSchema(vec![cov, mapq]);
    let missing_dim = plan_bed_schema(&sources, &grouped, &Config::default())
        .err()
        .unwrap()
        .to_string();
    assert!(
        missing_dim.contains("multiple BED columns"),
        "{missing_dim}"
    );
    assert!(missing_dim.contains("rank 2"), "{missing_dim}");
    assert!(missing_dim.contains("column dimension"), "{missing_dim}");

    let mut cov = BedColumnSpec::named("cov", Dtype::U16);
    cov.track_name = Some("stats".into());
    let mut score = BedColumnSpec::named("score", Dtype::F32);
    score.track_name = Some("stats".into());
    let mixed = BedSchema(vec![cov, score]);
    let mixed_dtype = plan_bed_schema(
        &sources,
        &mixed,
        &Config {
            column_dim: Some("metric".into()),
            ..Config::default()
        },
    )
    .err()
    .unwrap()
    .to_string();
    assert!(mixed_dtype.contains("BED columns"), "{mixed_dtype}");
    assert!(mixed_dtype.contains("track \"stats\""), "{mixed_dtype}");
    assert!(mixed_dtype.contains("share one dtype"), "{mixed_dtype}");
}

#[test]
fn unique_schema_rejects_duplicate_source_axis_labels() {
    let dir = TempDir::new().unwrap();
    let first = write_bed_bgzip_tabix(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let second = write_bed_bgzip_tabix(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["9"])],
    );
    let sources = [
        Source::labeled(first, "duplicate"),
        Source::labeled(second, "duplicate"),
    ];
    let error = plan_bed_schema(
        &sources,
        &BedSchema(vec![BedColumnSpec::named("cov", Dtype::U16)]),
        &Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .err()
    .expect("duplicate source-axis labels must fail")
    .to_string();
    assert!(error.contains("duplicate"), "{error}");
    assert!(error.contains("label"), "{error}");
}

#[test]
fn schema_execution_rejects_a_column_dimension_changed_after_planning() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "one",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(
        &sources,
        &BedSchema(vec![BedColumnSpec::named("cov", Dtype::U16)]),
        &Config::default(),
    )
    .unwrap();
    let (path, mut dataset) = create(dir.path(), "changed-config.pbz", 8);

    let error = execute_bed_schema_plan(
        &mut dataset,
        plan,
        genome(&[("chr1", 8)]),
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .err()
    .expect("execution must not change the planned rank")
    .to_string();
    assert!(error.contains("column dimension"), "{error}");
    assert!(error.contains("planning"), "{error}");
    assert!(track_names(&path).is_empty());
}

#[test]
fn schema_plan_rejects_maximum_index_without_panicking() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "one",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let sources = [Source::new(bed)];
    let schema = BedSchema(vec![BedColumnSpec::indexed(usize::MAX, Dtype::U16, "cov")]);

    let result = std::panic::catch_unwind(|| {
        plan_bed_schema(&sources, &schema, &Config::default())
            .err()
            .expect("usize::MAX must be rejected")
            .to_string()
    });
    let error = result.expect("maximum index must not panic");
    assert!(error.contains("BED column"), "{error}");
    assert!(error.contains("one-based"), "{error}");
}

#[test]
fn multi_column_mixed_dtype_roundtrip() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "score", "mask"],
        &[
            ("chr1", 0, 20, vec!["4", "1.5", "1"]),
            ("chr1", 20, 40, vec!["9", "2.5", "0"]),
        ],
    );
    let output = dir.path().join("out.pbz");

    // Two name selectors (default track names) + one index selector.
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("score", Dtype::F32),
        BedColumnSpec::indexed(5, Dtype::Bool, "mask"),
    ]);
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    execute_bed_schema_plan(
        output.as_path(),
        plan,
        genome(&[("chr1", 40)]),
        Config::default(),
    )
    .unwrap();

    let dataset = Dataset::open(&output).unwrap();
    let cov = dataset.read_region::<i32>("cov", &chr1()).unwrap();
    assert_eq!(cov.shape(), [40]);
    assert!(cov.iter().take(20).all(|&v| v == 4));
    assert!(cov.iter().skip(20).all(|&v| v == 9));

    let score = dataset.read_region::<f32>("score", &chr1()).unwrap();
    assert!(score.iter().take(20).all(|&v| v == 1.5));
    assert!(score.iter().skip(20).all(|&v| v == 2.5));

    let mask = dataset.read_region::<bool>("mask", &chr1()).unwrap();
    assert!(mask.iter().take(20).all(|&v| v));
    assert!(mask.iter().skip(20).all(|&v| !v));
}

#[test]
fn multi_source_matrix_import_roundtrip() {
    let dir = TempDir::new().unwrap();
    let a = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage", "score"],
        &[("chr1", 0, 8, vec!["4", "1.5"])],
    );
    let b = write_bed_bgzip_tabix(
        dir.path(),
        "b",
        &["chrom", "start", "end", "coverage", "score"],
        &[("chr1", 0, 8, vec!["9", "2.5"])],
    );
    let output = dir.path().join("matrix.pbz");
    from_bed_matrix(
        output.as_path(),
        &[Source::labeled(a, "A"), Source::labeled(b, "B")],
        genome(&[("chr1", 8)]),
        &BedImportOptions::default(),
        Config {
            chunk_size: Some(4),
            column_chunk_size: Some(1),
            shard_size: Some(4),
            shard_column_size: Some(2),
            workers: 2,
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .unwrap();
    let dataset = Dataset::open(&output).unwrap();
    let coverage = dataset
        .read_region::<u8>("coverage", &chr1())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let score = dataset
        .read_region::<f32>("score", &chr1())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert!(
        coverage
            .rows()
            .into_iter()
            .all(|row| row.to_vec() == vec![4, 9])
    );
    assert!(
        score
            .rows()
            .into_iter()
            .all(|row| row.to_vec() == vec![1.5, 2.5])
    );
}

/// One source with an explicit `column_dim` (no second source) must still
/// build a rank-2, 1-column track through the `PerField` path: a single
/// labeled source with an explicit column dimension makes a one-column 2D
/// track, so the routing keys on that, not on source count.
#[test]
fn single_source_with_column_dim_is_one_column_2d() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let output = dir.path().join("single_column_dim.pbz");
    from_bed_matrix(
        output.as_path(),
        &[Source::new(bed)],
        genome(&[("chr1", 8)]),
        &BedImportOptions {
            fields: Some(vec!["cov".into()]),
            ..BedImportOptions::default()
        },
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .unwrap();

    let dataset = Dataset::open(&output).unwrap();
    assert_eq!(dataset.variables()["cov"].dims, ["position", "sample"]);
    let cov = dataset
        .read_region::<u8>("cov", &chr1())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert!(cov.rows().into_iter().all(|row| row.to_vec() == vec![4]));
}

/// Import `cov` (int32) and `flag` (bool) over a 40-base chr1 whose records
/// cover only [0,10) and [20,30).
fn import_gapped(dir: &Path, name: &str, cov_fill: Option<Fill>) -> Dataset {
    let bed = write_bed_bgzip_tabix(
        dir,
        name,
        &["chrom", "start", "end", "cov", "flag"],
        &[
            ("chr1", 0, 10, vec!["5", "1"]),
            ("chr1", 20, 30, vec!["7", "0"]),
        ],
    );
    let output = dir.join(format!("{name}.pbz"));
    let mut cov = BedColumnSpec::named("cov", Dtype::I32);
    cov.fill = cov_fill;
    let schema = BedSchema(vec![cov, BedColumnSpec::named("flag", Dtype::Bool)]);
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    execute_bed_schema_plan(
        output.as_path(),
        plan,
        genome(&[("chr1", 40)]),
        Config::default(),
    )
    .unwrap();
    Dataset::open(&output).unwrap()
}

#[test]
fn uncovered_positions_read_as_the_null_by_default() {
    let dir = TempDir::new().unwrap();
    let dataset = import_gapped(dir.path(), "na", None);
    let cov = dataset.read_region::<i32>("cov", &chr1()).unwrap();
    assert_eq!(cov.shape(), [40]);
    assert!(cov.iter().take(10).all(|&v| v == 5));
    assert!(cov.iter().skip(10).take(10).all(|&v| v == i32::MAX));
    assert!(cov.iter().skip(20).take(10).all(|&v| v == 7));
    assert!(cov.iter().skip(30).all(|&v| v == i32::MAX));
    assert_eq!(
        dataset.variables()["cov"].metadata["attributes"]["_FillValue"],
        i32::MAX
    );

    let flag = dataset.read_region::<bool>("flag", &chr1()).unwrap();
    assert!(flag.iter().take(10).all(|&v| v));
    assert!(flag.iter().skip(10).all(|&v| !v));
    assert!(dataset.variables()["flag"].metadata["attributes"]["_FillValue"].is_null());
}

#[test]
fn uncovered_positions_read_as_a_fill_of_zero() {
    let dir = TempDir::new().unwrap();
    let fill = Fill::Value(serde_json::json!(0));
    let dataset = import_gapped(dir.path(), "zero", Some(fill));
    let cov = dataset.read_region::<i32>("cov", &chr1()).unwrap();
    assert!(cov.iter().skip(10).take(10).all(|&v| v == 0));
    assert!(cov.iter().skip(30).all(|&v| v == 0));
    assert!(dataset.variables()["cov"].metadata["attributes"]["_FillValue"].is_null());
}

#[test]
fn a_value_equal_to_the_null_is_refused() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "max",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 10, vec!["255"])],
    );
    let schema = BedSchema(vec![BedColumnSpec::named("cov", Dtype::U8)]);
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    let Err(error) = execute_bed_schema_plan(
        dir.path().join("max.pbz").as_path(),
        plan,
        genome(&[("chr1", 10)]),
        Config::default(),
    ) else {
        panic!("a cell equal to the null must be refused");
    };
    assert!(error.to_string().contains("null of uint8"), "{error}");
}

#[test]
fn population_failure_leaves_all_tracks_unpublished() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "invalid",
        &["chrom", "start", "end", "cov", "flag"],
        &[("chr1", 0, 10, vec!["not-an-int", "1"])],
    );
    let (path, mut dataset) = create(dir.path(), "out.pbz", 10);
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);

    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    let result = execute_bed_schema_plan(
        &mut dataset,
        plan,
        genome(&[("chr1", 10)]),
        Config::default(),
    );

    assert!(result.is_err());
    assert!(track_names(&path).is_empty());
}

#[test]
fn reordered_headers_are_rejected() {
    let dir = TempDir::new().unwrap();
    let a = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage", "score"],
        &[("chr1", 0, 8, vec!["4", "1.5"])],
    );
    let b = write_bed_bgzip_tabix(
        dir.path(),
        "b",
        &["chrom", "start", "end", "score", "coverage"],
        &[("chr1", 0, 8, vec!["2.5", "9"])],
    );
    let (path, mut dataset) = create(dir.path(), "out.pbz", 8);
    let sources = [Source::new(a), Source::new(b)];
    let schema_error = plan_bed_schema(
        &sources,
        &BedSchema(vec![
            BedColumnSpec::named("coverage", Dtype::U16),
            BedColumnSpec::named("score", Dtype::F32),
        ]),
        &Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .err()
    .expect("schema planning must reject reordered source headers")
    .to_string();
    assert!(
        schema_error.contains("BED column header differs"),
        "{schema_error}"
    );
    let result = from_bed_matrix(
        &mut dataset,
        &sources,
        genome(&[("chr1", 8)]),
        &BedImportOptions::default(),
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    );
    assert!(result.is_err());
    assert!(track_names(&path).is_empty());
}

#[test]
fn headerless_bed4_cohort_roundtrip() {
    let dir = TempDir::new().unwrap();
    let a = write_headerless_bed_bgzip_tabix(dir.path(), "s1", &[("chr1", 0, 8, vec!["4"])]);
    let b = write_headerless_bed_bgzip_tabix(dir.path(), "s2", &[("chr1", 0, 8, vec!["9"])]);
    let output = dir.path().join("out.pbz");
    let sources = [Source::new(a), Source::new(b)];

    let error = from_bed_matrix(
        output.as_path(),
        &sources,
        genome(&[("chr1", 8)]),
        &BedImportOptions::default(),
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .err()
    .expect("headerless input without a track name must fail")
    .to_string();
    assert!(error.contains("track name is required"), "{error}");

    let options = BedImportOptions {
        track: Some("depth".into()),
        ..BedImportOptions::default()
    };
    from_bed_matrix(
        output.as_path(),
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .unwrap();

    let depth = Dataset::open(&output)
        .unwrap()
        .read_region::<u8>("depth", &chr1())
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert!(
        depth
            .rows()
            .into_iter()
            .all(|row| row.to_vec() == vec![4, 9])
    );
}

#[test]
fn bed3_imports_bool_presence() {
    let dir = TempDir::new().unwrap();
    let bed = write_headerless_bed_bgzip_tabix(dir.path(), "s1", &[("chr1", 2, 5, vec![])]);
    let output = dir.path().join("out.pbz");
    let sources = [Source::new(bed)];
    let options = BedImportOptions {
        track: Some("callable".into()),
        ..BedImportOptions::default()
    };
    from_bed_matrix(
        output.as_path(),
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config::default(),
    )
    .unwrap();

    let mask = Dataset::open(&output)
        .unwrap()
        .read_region::<bool>("callable", &chr1())
        .unwrap()
        .into_raw_vec_and_offset()
        .0;
    assert_eq!(
        mask,
        vec![false, false, true, true, true, false, false, false]
    );
}

#[test]
fn wide_track_rejects_word_bool_field_before_creating_tracks() {
    let dir = TempDir::new().unwrap();
    // Mirrors a FIRE pileup: numeric fields plus a true/false word column.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "score", "is_local_max"],
        &[("chr1", 0, 8, vec!["1.5", "false"])],
    );
    let (path, mut dataset) = create(dir.path(), "out.pbz", 8);
    let sources = [Source::new(bed)];
    let options = BedImportOptions {
        track: Some("stats".into()),
        ..BedImportOptions::default()
    };
    let result = from_bed_matrix(
        &mut dataset,
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config {
            column_dim: Some("metric".into()),
            ..Config::default()
        },
    );
    let error = match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("wide import with a word-bool field should fail"),
    };
    assert!(error.contains("is_local_max"), "unexpected error: {error}");
    assert!(error.contains("BED column(s)"), "unexpected error: {error}");
    assert!(
        error.contains("`--field` selection"),
        "unexpected error: {error}"
    );
    assert!(!error.contains("field(s)"), "unexpected error: {error}");
    assert!(track_names(&path).is_empty());
}
