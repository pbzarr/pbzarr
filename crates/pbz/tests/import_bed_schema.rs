use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use pbzarr::io::Dtype;
use pbzarr::{PbzStore, Region};
use tempfile::TempDir;

fn require_htslib() {
    for bin in ["bgzip", "tabix"] {
        let available = Command::new(bin)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        assert!(
            available,
            "{bin} is required by the BED schema CLI regressions; run them with `pixi run test`"
        );
    }
}

fn write_bed(
    dir: &Path,
    name: &str,
    header: &[&str],
    rows: &[(&str, u64, u64, &[&str])],
) -> PathBuf {
    let bed = dir.join(format!("{name}.bed"));
    let mut file = File::create(&bed).unwrap();
    writeln!(file, "#{}", header.join("\t")).unwrap();
    for (contig, start, end, values) in rows {
        write!(file, "{contig}\t{start}\t{end}").unwrap();
        for value in *values {
            write!(file, "\t{value}").unwrap();
        }
        writeln!(file).unwrap();
    }
    drop(file);

    assert!(
        Command::new("bgzip")
            .arg("-f")
            .arg(&bed)
            .status()
            .unwrap()
            .success()
    );
    let compressed = dir.join(format!("{name}.bed.gz"));
    assert!(
        Command::new("tabix")
            .args(["-f", "-p", "bed"])
            .arg(&compressed)
            .status()
            .unwrap()
            .success()
    );
    compressed
}

fn write_genome(dir: &Path) -> PathBuf {
    let genome = dir.join("genome.fai");
    std::fs::write(&genome, "chr1\t8\n").unwrap();
    genome
}

fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

fn import_args(
    output: &Path,
    genome: &Path,
    schema: &Path,
    column_dim: &str,
    sources: impl IntoIterator<Item = OsString>,
) -> Vec<OsString> {
    let mut args = vec![
        "import".into(),
        "bed".into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--genome".into(),
        genome.as_os_str().to_owned(),
        "--schema".into(),
        schema.as_os_str().to_owned(),
        "--column-dim".into(),
        column_dim.into(),
        "--infer-rows".into(),
        "all".into(),
    ];
    args.extend(sources);
    args
}

fn labeled(path: &Path, label: &str) -> OsString {
    format!("{}:{label}", path.display()).into()
}

#[test]
fn grouped_schema_one_source_creates_an_ordered_bed_column_axis() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "wide",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 4, &["4", "60"]), ("chr1", 4, 8, &["9", "30"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "mapq\tstats\tuint16\ncov\tstats\tuint16\n").unwrap();
    let output = dir.path().join("grouped.pbz");

    let command = run_pbz(import_args(
        &output,
        &genome,
        &schema,
        "metric",
        [bed.into_os_string()],
    ));
    assert!(
        command.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&command.stderr)
    );

    let store = PbzStore::open(&output).unwrap();
    let track = store.track("stats").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.dtype(), Dtype::U16);
    assert_eq!(track.column_dim(), Some("metric"));
    assert_eq!(
        track.column_labels().unwrap(),
        vec!["mapq".to_owned(), "cov".to_owned()]
    );
    let region = Region {
        contig: store.genome_for("stats").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let values = track.read_region::<u16>(&region).unwrap();
    assert_eq!(values.shape(), &[8, 2]);
    assert_eq!(
        values.iter().copied().collect::<Vec<_>>(),
        vec![60, 4, 60, 4, 60, 4, 60, 4, 30, 9, 30, 9, 30, 9, 30, 9]
    );
}

#[test]
fn unique_schema_two_sources_infers_across_sources_and_keeps_source_labels() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let tumor = write_bed(
        dir.path(),
        "tumor",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["4", "60"])],
    );
    let normal = write_bed(
        dir.path(),
        "normal",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["70000", "30"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "mapq\tmapping\ncov\tcoverage\n").unwrap();
    let output = dir.path().join("unique.pbz");

    let command = run_pbz(import_args(
        &output,
        &genome,
        &schema,
        "sample",
        [labeled(&tumor, "tumor"), labeled(&normal, "normal")],
    ));
    assert!(
        command.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&command.stderr)
    );

    let store = PbzStore::open(&output).unwrap();
    for (name, dtype) in [("mapping", Dtype::U8), ("coverage", Dtype::U32)] {
        let track = store.track(name).unwrap();
        assert_eq!(track.rank(), 2);
        assert_eq!(track.dtype(), dtype);
        assert_eq!(track.column_dim(), Some("sample"));
        assert_eq!(
            track.column_labels().unwrap(),
            vec!["tumor".to_owned(), "normal".to_owned()]
        );
    }
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 8,
    };
    let coverage = store
        .track("coverage")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    assert_eq!(coverage.shape(), &[8, 2]);
    assert_eq!(
        coverage.iter().copied().collect::<Vec<_>>(),
        [4, 70_000].repeat(8)
    );
    let mapping = store
        .track("mapping")
        .unwrap()
        .read_region::<u8>(&region)
        .unwrap();
    assert_eq!(mapping.shape(), &[8, 2]);
    assert_eq!(
        mapping.iter().copied().collect::<Vec<_>>(),
        [60, 30].repeat(8)
    );
}

#[test]
fn grouped_schema_two_sources_fails_preflight_without_creating_output() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let first = write_bed(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["4", "60"])],
    );
    let second = write_bed(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["9", "30"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "cov\tstats\tuint16\nmapq\tstats\tuint16\n").unwrap();
    let output = dir.path().join("must-not-exist.pbz");

    let command = run_pbz(import_args(
        &output,
        &genome,
        &schema,
        "metric",
        [labeled(&first, "a"), labeled(&second, "b")],
    ));
    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
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
    assert!(
        !output.exists(),
        "schema preflight created the output store"
    );
}

#[test]
fn unique_schema_two_sources_without_column_dimension_fails_before_output_creation() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let first = write_bed(
        dir.path(),
        "first",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, &["4"])],
    );
    let second = write_bed(
        dir.path(),
        "second",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, &["9"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "cov\tcoverage\tuint16\n").unwrap();
    let output = dir.path().join("missing-axis.pbz");

    let command = run_pbz([
        "import".into(),
        "bed".into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--genome".into(),
        genome.into_os_string(),
        "--schema".into(),
        schema.into_os_string(),
        labeled(&first, "a"),
        labeled(&second, "b"),
    ]);
    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
    assert!(error.contains("multiple input sources"), "{error}");
    assert!(error.contains("BED column track"), "{error}");
    assert!(error.contains("rank 2 (position, source)"), "{error}");
    assert!(error.contains("column dimension name"), "{error}");
    assert!(
        !output.exists(),
        "schema preflight created the output store"
    );
}

#[test]
fn out_of_range_numeric_schema_selector_reports_the_original_one_based_column() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "five-columns",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["4", "60"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "6\tmissing\tuint16\n").unwrap();
    let output = dir.path().join("out-of-range.pbz");

    let command = run_pbz([
        "import".into(),
        "bed".into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--genome".into(),
        genome.into_os_string(),
        "--schema".into(),
        schema.into_os_string(),
        bed.into_os_string(),
    ]);

    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
    assert!(
        error.contains("BED column 6 (one-based) is outside the 5-column input"),
        "{error}"
    );
    assert!(!error.contains("BED column 5 is outside"), "{error}");
    assert!(
        !output.exists(),
        "selector preflight created the output store"
    );
}

#[test]
fn out_of_range_numeric_schema_selector_during_inference_is_one_based() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "five-columns-inference",
        &["chrom", "start", "end", "cov", "mapq"],
        &[("chr1", 0, 8, &["4", "60"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, "6\tmissing\n").unwrap();
    let output = dir.path().join("out-of-range-inference.pbz");

    let command = run_pbz([
        "import".into(),
        "bed".into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--genome".into(),
        genome.into_os_string(),
        "--schema".into(),
        schema.into_os_string(),
        bed.into_os_string(),
    ]);

    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
    assert!(
        error.contains("BED column 6 (one-based) is outside the 5-column input"),
        "{error}"
    );
    assert!(!error.contains("no column 5"), "{error}");
    assert!(
        !output.exists(),
        "inference preflight created the output store"
    );
}

#[test]
fn overflowing_numeric_schema_selector_is_not_treated_as_a_header_name() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "overflow-selector",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, &["4"])],
    );
    let genome = write_genome(dir.path());
    let selector = format!("{}0", usize::MAX);
    let schema = dir.path().join("schema.tsv");
    std::fs::write(&schema, format!("{selector}\tmissing\tuint16\n")).unwrap();
    let output = dir.path().join("overflow.pbz");

    let command = run_pbz([
        "import".into(),
        "bed".into(),
        "--output".into(),
        output.as_os_str().to_owned(),
        "--genome".into(),
        genome.into_os_string(),
        "--schema".into(),
        schema.into_os_string(),
        bed.into_os_string(),
    ]);

    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
    assert!(
        error.contains(&format!(
            "invalid numeric BED-column selector {selector:?}: value overflows usize"
        )),
        "{error}"
    );
    assert!(!error.contains("not in header"), "{error}");
    assert!(
        !output.exists(),
        "selector preflight created the output store"
    );
}

#[test]
fn invalid_schema_target_track_names_fail_before_output_creation() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "invalid-target",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, &["4"])],
    );
    let genome = write_genome(dir.path());

    for (case, invalid) in ["__axis", "...", "nested/track"].into_iter().enumerate() {
        let schema = dir.path().join(format!("target-{case}.tsv"));
        std::fs::write(&schema, format!("cov\t{invalid}\tuint16\n")).unwrap();
        let output = dir.path().join(format!("invalid-target-{case}.pbz"));
        let command = run_pbz([
            "import".into(),
            "bed".into(),
            "--output".into(),
            output.as_os_str().to_owned(),
            "--genome".into(),
            genome.as_os_str().to_owned(),
            "--schema".into(),
            schema.into_os_string(),
            bed.as_os_str().to_owned(),
        ]);

        assert!(!command.status.success(), "target {invalid:?} was accepted");
        let error = String::from_utf8_lossy(&command.stderr);
        assert!(error.contains("target track name"), "{error}");
        assert!(error.contains("invalid PBZ/Zarr node name"), "{error}");
        assert!(error.contains(&format!("{invalid:?}")), "{error}");
        assert!(!output.exists(), "target preflight created {output:?}");
    }
}

#[test]
fn invalid_schema_column_dimension_names_fail_before_output_creation() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "invalid-dimension",
        &["chrom", "start", "end", "cov"],
        &[("chr1", 0, 8, &["4"])],
    );
    let genome = write_genome(dir.path());
    let schema = dir.path().join("dimension-schema.tsv");
    std::fs::write(&schema, "cov\tcoverage\tuint16\n").unwrap();

    for (case, invalid) in ["__axis", "...", "nested/axis"].into_iter().enumerate() {
        let output = dir.path().join(format!("invalid-dimension-{case}.pbz"));
        let command = run_pbz(import_args(
            &output,
            &genome,
            &schema,
            invalid,
            [bed.as_os_str().to_owned()],
        ));

        assert!(
            !command.status.success(),
            "dimension {invalid:?} was accepted"
        );
        let error = String::from_utf8_lossy(&command.stderr);
        assert!(error.contains("column dimension name"), "{error}");
        assert!(error.contains("invalid PBZ/Zarr node name"), "{error}");
        assert!(error.contains(&format!("{invalid:?}")), "{error}");
        assert!(!output.exists(), "dimension preflight created {output:?}");
    }
}

#[test]
fn emit_schema_bed3_error_calls_values_bed_columns() {
    require_htslib();
    let dir = TempDir::new().unwrap();
    let bed = write_bed(
        dir.path(),
        "regions",
        &["chrom", "start", "end"],
        &[("chr1", 0, 8, &[])],
    );

    let command = run_pbz([
        "import".into(),
        "bed".into(),
        "--emit-schema".into(),
        bed.into_os_string(),
    ]);
    assert!(!command.status.success());
    let error = String::from_utf8_lossy(&command.stderr);
    assert!(error.contains("no BED value columns"), "{error}");
    assert!(!error.contains("field"), "{error}");
}

#[test]
fn schema_help_describes_unique_and_grouped_target_tracks() {
    let command = run_pbz(["import".into(), "bed".into(), "--help".into()]);
    assert!(command.status.success());
    let help = String::from_utf8_lossy(&command.stdout);
    assert!(help.contains("BED columns to target tracks"), "{help}");
    assert!(help.contains("Unique target"), "{help}");
    assert!(help.contains("single BED-column-axis track"), "{help}");
    assert!(
        help.contains("BED column, target track, inferred dtype"),
        "{help}"
    );
    assert!(help.contains("-c metric --schema schema.tsv"), "{help}");
    assert!(!help.contains("own scalar track"), "{help}");
}
