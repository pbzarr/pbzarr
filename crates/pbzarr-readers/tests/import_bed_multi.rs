//! End-to-end: synthesize multi-column bgzipped+tabix BEDs and import them via
//! the schema generation (`plan_bed_schema`/`execute_bed_schema_plan`) and
//! `from_bed_matrix`, then read the tracks back.

mod common;

use ndarray::{Ix1, Ix2};
use pbzarr::genome::{Contig, Genome};
use pbzarr::import::Config;
use pbzarr::io::Dtype;
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{
    BedColumnSpec, BedImportOptions, BedSchema, Source, execute_bed_schema_plan, from_bed_matrix,
    plan_bed_schema,
};
use tempfile::TempDir;

use common::{write_bed_bgzip_tabix, write_headerless_bed_bgzip_tabix};

fn genome(cs: &[(&str, u64)]) -> Genome {
    Genome::new(
        cs.iter()
            .map(|(n, l)| Contig {
                name: (*n).to_owned(),
                length: *l,
            })
            .collect(),
    )
    .unwrap()
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
    let mut store = PbzStore::create(output).unwrap();
    execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 8)]), Config::default()).unwrap();

    let region = Region {
        contig: store.genome_for("first").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    for (track_name, expected) in [("first", 4u16), ("second", 9u16)] {
        let values = store
            .track(track_name)
            .unwrap()
            .read_region::<u16>(&region)
            .unwrap();
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
    let mut store = PbzStore::create(&output).unwrap();
    execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 8)]), config).unwrap();

    let track = store.track("stats").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.dtype(), Dtype::U16);
    assert_eq!(track.column_dim(), Some("BED column"));
    assert_eq!(track.columns_count().unwrap(), 2);
    assert_eq!(
        track.column_labels().unwrap(),
        vec!["mapq".to_owned(), "cov".to_owned()]
    );

    let region = Region {
        contig: store.genome_for("stats").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let values = track
        .read_region::<u16>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (index, row) in values.rows().into_iter().enumerate() {
        assert_eq!(
            row.to_vec(),
            if index < 4 { vec![60, 4] } else { vec![30, 9] }
        );
    }

    let metadata = std::fs::read_to_string(output.join("stats").join("zarr.json")).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata).unwrap();
    assert_eq!(
        metadata["attributes"]["perbase:description"],
        "Quality statistics"
    );
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
    let mut store = PbzStore::create(dir.path().join("unique.pbz")).unwrap();
    execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 8)]), config).unwrap();

    let region = Region {
        contig: store.genome_for("mapq").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    for (track_name, expected) in [("mapq", vec![60, 30]), ("cov", vec![4, 9])] {
        let track = store.track(track_name).unwrap();
        assert_eq!(track.rank(), 2);
        assert_eq!(track.dtype(), Dtype::U16);
        assert_eq!(track.column_dim(), Some("sample"));
        assert_eq!(
            track.column_labels().unwrap(),
            vec!["tumor".to_owned(), "normal".to_owned()]
        );
        let values = track
            .read_region::<u16>(&region)
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

    let store = PbzStore::create(dir.path().join("rank3.pbz")).unwrap();
    assert!(store.track_names().next().is_none());
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
    let mut store = PbzStore::create(dir.path().join("changed-config.pbz")).unwrap();

    let error = execute_bed_schema_plan(
        &mut store,
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
    assert!(store.track_names().next().is_none());
}

#[test]
fn schema_plan_rejects_invalid_column_dimension_names() {
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "one",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, vec!["4"])],
    );
    let sources = [Source::new(bed)];
    let schema = BedSchema(vec![BedColumnSpec::named("cov", Dtype::U16)]);

    for invalid in ["", "   ", "sample/group", "position", "values"] {
        let error = plan_bed_schema(
            &sources,
            &schema,
            &Config {
                column_dim: Some(invalid.into()),
                ..Config::default()
            },
        )
        .err()
        .unwrap_or_else(|| panic!("column dimension {invalid:?} must fail"))
        .to_string();
        assert!(error.contains("column dimension"), "{error}");
        assert!(error.contains("invalid"), "{error}");
    }
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
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();

    // Two name selectors (default track names) + one index selector.
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("score", Dtype::F32),
        BedColumnSpec::indexed(5, Dtype::Bool, "mask"),
    ]);
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 40)]), Config::default()).unwrap();

    // Every track owns an identical genome; contig ids share the same index.
    let g = store.genome_for("cov").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 40,
    };

    let cov = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(cov.iter().take(20).all(|&v| v == 4));
    assert!(cov.iter().skip(20).all(|&v| v == 9));

    let score = store
        .track("score")
        .unwrap()
        .read_region::<f32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(score.iter().take(20).all(|&v| v == 1.5));
    assert!(score.iter().skip(20).all(|&v| v == 2.5));

    let mask = store
        .track("mask")
        .unwrap()
        .read_region::<bool>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
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
    let mut store = PbzStore::create(dir.path().join("matrix.pbz")).unwrap();
    from_bed_matrix(
        &mut store,
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
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let coverage = store
        .track("coverage")
        .unwrap()
        .read_region::<u8>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    let score = store
        .track("score")
        .unwrap()
        .read_region::<f32>(&region)
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
    let mut store = PbzStore::create(dir.path().join("single_column_dim.pbz")).unwrap();
    from_bed_matrix(
        &mut store,
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

    let track = store.track("cov").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.columns_count().unwrap(), 1);
    assert_eq!(track.column_dim(), Some("sample"));

    let region = Region {
        contig: store.genome_for("cov").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let cov = store
        .track("cov")
        .unwrap()
        .read_region::<u8>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert!(cov.rows().into_iter().all(|row| row.to_vec() == vec![4]));
}

#[test]
fn uncovered_positions_read_back_as_zero() {
    let dir = TempDir::new().unwrap();
    // chr1 is length 40 but records cover only [0,10) and [20,30); the gap
    // [10,20) and the tail [30,40) are never written and must read as zero.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "flag"],
        &[
            ("chr1", 0, 10, vec!["5", "1"]),
            ("chr1", 20, 30, vec!["7", "0"]),
        ],
    );
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);
    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 40)]), Config::default()).unwrap();

    let g = store.genome_for("cov").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 40,
    };

    let cov = store
        .track("cov")
        .unwrap()
        .read_region::<i32>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(cov.iter().take(10).all(|&v| v == 5));
    assert!(
        cov.iter().skip(10).take(10).all(|&v| v == 0),
        "gap reads as zero"
    );
    assert!(cov.iter().skip(20).take(10).all(|&v| v == 7));
    assert!(cov.iter().skip(30).all(|&v| v == 0), "tail reads as zero");

    let flag = store
        .track("flag")
        .unwrap()
        .read_region::<bool>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(flag.iter().take(10).all(|&v| v));
    assert!(
        flag.iter().skip(10).all(|&v| !v),
        "uncovered + true-zero both read false"
    );
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
    let path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    let schema = BedSchema(vec![
        BedColumnSpec::named("cov", Dtype::I32),
        BedColumnSpec::named("flag", Dtype::Bool),
    ]);

    let sources = [Source::new(bed)];
    let plan = plan_bed_schema(&sources, &schema, &Config::default()).unwrap();
    let result =
        execute_bed_schema_plan(&mut store, plan, genome(&[("chr1", 10)]), Config::default());

    assert!(result.is_err());
    assert!(store.track("cov").is_none());
    assert!(store.track("flag").is_none());
    let reopened = PbzStore::open(path).unwrap();
    assert!(reopened.track("cov").is_none());
    assert!(reopened.track("flag").is_none());
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
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
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
    assert!(store.track_names().next().is_none());
    let result = from_bed_matrix(
        &mut store,
        &sources,
        genome(&[("chr1", 8)]),
        &BedImportOptions::default(),
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    );
    assert!(result.is_err());
}

#[test]
fn headerless_bed4_cohort_roundtrip() {
    let dir = TempDir::new().unwrap();
    let a = write_headerless_bed_bgzip_tabix(dir.path(), "s1", &[("chr1", 0, 8, vec!["4"])]);
    let b = write_headerless_bed_bgzip_tabix(dir.path(), "s2", &[("chr1", 0, 8, vec!["9"])]);
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let sources = [Source::new(a), Source::new(b)];

    let error = from_bed_matrix(
        &mut store,
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
        &mut store,
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config {
            column_dim: Some("sample".into()),
            ..Config::default()
        },
    )
    .unwrap();

    let g = store.genome_for("depth").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let depth = store
        .track("depth")
        .unwrap()
        .read_region::<u8>(&reg)
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
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let sources = [Source::new(bed)];
    let options = BedImportOptions {
        track: Some("callable".into()),
        ..BedImportOptions::default()
    };
    from_bed_matrix(
        &mut store,
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config::default(),
    )
    .unwrap();

    let g = store.genome_for("callable").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let mask = store
        .track("callable")
        .unwrap()
        .read_region::<bool>(&reg)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert_eq!(
        mask.to_vec(),
        vec![false, false, true, true, true, false, false, false]
    );
}

#[test]
fn single_source_wide_track_with_joint_inference() {
    let dir = TempDir::new().unwrap();
    // cov fits u8 alone; mapq's 300 forces the joint dtype up to u16.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "cov", "mapq"],
        &[
            ("chr1", 0, 4, vec!["4", "60"]),
            ("chr1", 4, 8, vec!["9", "300"]),
        ],
    );
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let sources = [Source::new(bed)];
    let options = BedImportOptions {
        track: Some("stats".into()),
        ..BedImportOptions::default()
    };
    from_bed_matrix(
        &mut store,
        &sources,
        genome(&[("chr1", 8)]),
        &options,
        Config {
            column_dim: Some("field".into()),
            ..Config::default()
        },
    )
    .unwrap();

    let g = store.genome_for("stats").unwrap();
    let reg = Region {
        contig: g.id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let stats = store
        .track("stats")
        .unwrap()
        .read_region::<u16>(&reg)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    for (i, row) in stats.rows().into_iter().enumerate() {
        let expected = if i < 4 { vec![4, 60] } else { vec![9, 300] };
        assert_eq!(row.to_vec(), expected);
    }
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
    let mut store = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let sources = [Source::new(bed)];
    let options = BedImportOptions {
        track: Some("stats".into()),
        ..BedImportOptions::default()
    };
    let result = from_bed_matrix(
        &mut store,
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
    assert!(store.track("stats").is_none());
}
