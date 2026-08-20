use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ndarray::{Array1, Array2};
use noodles_bgzf as bgzf;
use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, TrackConfig};
use tempfile::TempDir;

fn genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 10,
        },
        Contig {
            name: "chr2".into(),
            length: 6,
        },
    ])
    .unwrap()
}

fn build_store(dir: &Path) -> PathBuf {
    let path = dir.join("view.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", genome(), TrackConfig::new(Dtype::I32))
        .unwrap();
    store
        .create_track(
            "af",
            genome(),
            TrackConfig::new(Dtype::F32)
                .columns(vec!["s1".into(), "s2".into()])
                .column_dim("sample"),
        )
        .unwrap();
    let reference = genome();
    let depth = store.track("depth").unwrap();
    let region = reference.resolve(&"chr1".parse().unwrap()).unwrap();
    depth
        .write_region(
            &region,
            Array1::from(vec![5i32, 5, 5, 7, 7, 0, 0, 0, 0, 0]).into_dyn(),
        )
        .unwrap();
    let af = store.track("af").unwrap();
    let region = reference.resolve(&"chr1:2-6".parse().unwrap()).unwrap();
    af.write_region(
        &region,
        Array2::from_shape_vec((4, 2), vec![0.5f32, 1.0, 0.5, 1.0, 0.25, 1.0, 0.25, 1.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    path
}

fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout_of(output: &Output) -> String {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn integer_track_streams_collapsed_rows_in_genome_order() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());

    let ambiguous = run_pbz(["view".into(), store.as_os_str().to_owned()]);
    assert!(!ambiguous.status.success());
    let error = String::from_utf8_lossy(&ambiguous.stderr);
    assert!(error.contains("store has 2 tracks"), "{error}");

    let output = run_pbz(["view".into(), store.into_os_string(), "depth".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\tdepth\n\
         chr1\t0\t3\t5\n\
         chr1\t3\t5\t7\n\
         chr1\t5\t10\t0\n\
         chr2\t0\t6\t0\n"
    );
}

#[test]
fn float_track_skips_all_missing_positions() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["view".into(), store.into_os_string(), "af".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t2\t4\t0.5\t1.0\n\
         chr1\t4\t6\t0.25\t1.0\n"
    );
}

#[test]
fn column_subset_and_no_header() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz([
        "view".into(),
        "--no-header".into(),
        "-c".into(),
        "s2".into(),
        store.into_os_string(),
        "af".into(),
    ]);
    assert_eq!(stdout_of(&output), "chr1\t2\t6\t1.0\n");
}

#[test]
fn gz_output_writes_bgzf_with_identical_content() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let gz = dir.path().join("out.bedgraph.gz");
    let plain = stdout_of(&run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "depth".into(),
    ]));
    let output = run_pbz([
        "view".into(),
        "-o".into(),
        gz.as_os_str().to_owned(),
        store.into_os_string(),
        "depth".into(),
    ]);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());

    let raw = std::fs::read(&gz).unwrap();
    assert_eq!(&raw[..2], &[0x1f, 0x8b], "not gzip-framed");
    let mut reader = bgzf::io::Reader::new(File::open(&gz).unwrap());
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    assert_eq!(text, plain);
}

#[test]
fn threads_option_does_not_change_output() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let default_run = stdout_of(&run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "depth".into(),
    ]));
    let capped = stdout_of(&run_pbz([
        "view".into(),
        "-t".into(),
        "1".into(),
        store.into_os_string(),
        "depth".into(),
    ]));
    assert_eq!(capped, default_run);
}
