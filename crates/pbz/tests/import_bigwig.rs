//! End-to-end BigWig imports using fixtures synthesized without external tools.
// One range per axis; the single-range reads are rank-one selections.
#![allow(clippy::single_range_in_vec_init)]

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use bigtools::BigWigWrite;
use bigtools::beddata::BedParserStreamingIterator;
use pbzarr::dataset::Dataset;
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

    let dataset = Dataset::open(&output).unwrap();
    assert_eq!(
        dataset.variables()["signal"].dims,
        vec!["position".to_owned()]
    );
    let values = dataset.read_raw::<f32>("signal", &[0..12]).unwrap();
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
            ]);
        } else {
            args.extend([labeled(&first, "left"), labeled(&second, "right")]);
        }
        stdout_of(&run_pbz(args));

        let dataset = Dataset::open(&output).unwrap();
        let column = if use_manifest { "context" } else { "sample" };
        assert_eq!(
            dataset.variables()["signal"].dims,
            vec!["position".to_owned(), column.to_owned()]
        );
        let labels = dataset.read_raw::<String>(column, &[0..2]).unwrap();
        assert_eq!(
            labels.iter().cloned().collect::<Vec<_>>(),
            ["left", "right"]
        );
        let values = dataset.read_raw::<f32>("signal", &[0..12, 0..2]).unwrap();
        assert_eq!(values.shape(), &[12, 2]);
        for position in 0..12 {
            let expected = if position < 6 { [0.5, 2.5] } else { [1.5, 3.5] };
            assert_eq!(values[[position, 0]], expected[0]);
            assert_eq!(values[[position, 1]], expected[1]);
        }

        let mean = run_pbz(["stat".into(), output.into_os_string(), "signal".into()]);
        assert_eq!(
            stdout_of(&mean),
            "#chrom\tstart\tend\tleft\tright\nchr1\t0\t12\t1\t3\n"
        );
    }
}
