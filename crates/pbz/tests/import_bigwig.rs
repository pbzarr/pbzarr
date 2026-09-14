//! End-to-end BigWig imports using fixtures synthesized without external tools.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use bigtools::BigWigWrite;
use bigtools::beddata::BedParserStreamingIterator;
use pbzarr::PbzStore;
use pbzarr::io::Dtype;
use tempfile::TempDir;

mod common;
use common::{run_pbz, stdout_of};

fn write_bigwig(dir: &Path, name: &str, intervals: &[(u32, u32, f32)]) -> PathBuf {
    let bedgraph = dir.join(format!("{name}.bedgraph"));
    let mut file = std::fs::File::create(&bedgraph).unwrap();
    for (start, end, value) in intervals {
        writeln!(file, "chr1\t{start}\t{end}\t{value}").unwrap();
    }
    drop(file);

    let path = dir.join(format!("{name}.bw"));
    let chromosomes = HashMap::from([("chr1".to_owned(), 12)]);
    let values = BedParserStreamingIterator::from_bedgraph_file(
        std::fs::File::open(bedgraph).unwrap(),
        false,
    );
    let mut writer = BigWigWrite::create_file(&path, chromosomes).unwrap();
    writer.options.channel_size = 0;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    writer.write(values, runtime).unwrap();
    path
}

fn import_args(output: &Path) -> Vec<OsString> {
    vec![
        "import".into(),
        "bigwig".into(),
        "-o".into(),
        output.as_os_str().to_owned(),
        "--track".into(),
        "signal".into(),
        "--no-progress".into(),
    ]
}

fn labeled(path: &Path, label: &str) -> OsString {
    format!("{}:{label}", path.display()).into()
}

#[test]
fn scalar_import_preserves_fractional_values_and_missing_positions() {
    let dir = TempDir::new().unwrap();
    let bw = write_bigwig(dir.path(), "sparse", &[(0, 4, 0.5), (6, 10, -1.5)]);
    let output = dir.path().join("signal.pbz");
    let mut args = import_args(&output);
    args.extend(["--chunk-size".into(), "4".into(), bw.into_os_string()]);
    stdout_of(&run_pbz(args));

    let store = PbzStore::open(&output).unwrap();
    let track = store.track("signal").unwrap();
    assert_eq!(track.dtype(), Dtype::F32);
    assert_eq!(track.rank(), 1);
    assert_eq!(track.chunk_size().unwrap(), 4);
    let region = track.genome().resolve(&"chr1".parse().unwrap()).unwrap();
    let values = track.read_region::<f32>(&region).unwrap();
    assert_eq!(values.shape(), &[12]);
    for (index, expected) in [
        0.5,
        0.5,
        0.5,
        0.5,
        f32::NAN,
        f32::NAN,
        -1.5,
        -1.5,
        -1.5,
        -1.5,
        f32::NAN,
        f32::NAN,
    ]
    .into_iter()
    .enumerate()
    {
        if expected.is_nan() {
            assert!(values[index].is_nan(), "position {index} must be missing");
        } else {
            assert_eq!(values[index], expected, "position {index}");
        }
    }

    let mean = run_pbz(["stat".into(), output.into_os_string(), "signal".into()]);
    assert_eq!(
        stdout_of(&mean),
        "#chrom\tstart\tend\tmean\nchr1\t0\t12\t-0.5\n"
    );
}

#[test]
fn labeled_cohorts_accept_positional_sources_and_a_manifest() {
    let dir = TempDir::new().unwrap();
    let first = write_bigwig(dir.path(), "first", &[(0, 6, 0.5), (6, 12, 1.5)]);
    let second = write_bigwig(dir.path(), "second", &[(0, 6, 2.5), (6, 12, 3.5)]);
    let manifest = dir.path().join("sources.tsv");
    std::fs::write(
        &manifest,
        format!("{}\tleft\n{}\tright\n", first.display(), second.display()),
    )
    .unwrap();

    for use_manifest in [false, true] {
        let output = dir.path().join(if use_manifest {
            "manifest.pbz"
        } else {
            "positional.pbz"
        });
        let mut args = import_args(&output);
        if use_manifest {
            args.extend([
                "--file-list".into(),
                manifest.as_os_str().to_owned(),
                "--column-dim".into(),
                "context".into(),
                "--chunk-size".into(),
                "4".into(),
                "--shard-size".into(),
                "8".into(),
                "--column-chunk-size".into(),
                "1".into(),
                "--shard-column-size".into(),
                "2".into(),
                "--scales".into(),
                "4".into(),
            ]);
        } else {
            args.extend([labeled(&first, "left"), labeled(&second, "right")]);
        }
        stdout_of(&run_pbz(args));

        let store = PbzStore::open(&output).unwrap();
        let track = store.track("signal").unwrap();
        assert_eq!(track.dtype(), Dtype::F32);
        assert_eq!(track.rank(), 2);
        assert_eq!(
            track.column_dim(),
            Some(if use_manifest { "context" } else { "sample" })
        );
        assert_eq!(track.column_labels().unwrap(), ["left", "right"]);
        let region = track.genome().resolve(&"chr1".parse().unwrap()).unwrap();
        let values = track.read_region::<f32>(&region).unwrap();
        assert_eq!(values.shape(), &[12, 2]);
        for position in 0..12 {
            let expected = if position < 6 { [0.5, 2.5] } else { [1.5, 3.5] };
            assert_eq!(values[[position, 0]], expected[0]);
            assert_eq!(values[[position, 1]], expected[1]);
        }
        if use_manifest {
            // The physical chunk is the outer shard. Twelve positions also
            // exercise its partial tail; --scales must publish the pyramid.
            assert_eq!(track.chunk_size().unwrap(), 8);
            assert!(output.join("signal/scales/4/mean/zarr.json").is_file());
        }

        let mean = run_pbz(["stat".into(), output.into_os_string(), "signal".into()]);
        assert_eq!(
            stdout_of(&mean),
            "#chrom\tstart\tend\tleft\tright\nchr1\t0\t12\t1\t3\n"
        );
    }
}

#[test]
fn dry_run_reports_float_schema_without_creating_a_store() {
    let dir = TempDir::new().unwrap();
    let bw = write_bigwig(dir.path(), "signal", &[(0, 12, 1.5)]);
    let cases = [
        (
            vec![bw.as_os_str().to_owned()],
            "track signal  float32  rank 1",
        ),
        (
            vec![labeled(&bw, "left"), labeled(&bw, "right")],
            "track signal  float32  rank 2  columns (sample): left, right",
        ),
        (
            vec![
                "--column-dim".into(),
                "context".into(),
                labeled(&bw, "single"),
            ],
            "track signal  float32  rank 2  columns (context): single",
        ),
    ];
    for (index, (sources, expected)) in cases.into_iter().enumerate() {
        let output = dir.path().join(format!("dry-run-{index}.pbz"));
        let mut args = import_args(&output);
        args.push("--dry-run".into());
        args.extend(sources);
        let stdout = stdout_of(&run_pbz(args));
        assert!(stdout.contains(expected), "{stdout}");
        assert!(
            stdout.contains("genome: 1 contig(s), 12 positions"),
            "{stdout}"
        );
        assert!(stdout.contains("tasks/worker"), "{stdout}");
        assert!(!output.exists(), "dry run created the output store");
    }
}
