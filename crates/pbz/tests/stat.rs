use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ndarray::{Array1, Array2};
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
    let path = dir.join("stat.pbz");
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
fn mean_defaults_to_one_row_per_chromosome() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["stat".into(), store.into_os_string(), "depth".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\tmean\n\
         chr1\t0\t10\t2.9\n\
         chr2\t0\t6\t0\n"
    );
}

#[test]
fn wide_2d_mean_skips_nan_and_names_samples() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz(["stat".into(), store.into_os_string(), "af".into()]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t0\t10\t0.375\t1\n\
         chr2\t0\t6\tNaN\tNaN\n"
    );
}

fn write_bed(dir: &Path, text: &str) -> PathBuf {
    let path = dir.join("regions.bed");
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn bed_regions_keep_order_including_duplicates_and_overlaps() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let bed = write_bed(dir.path(), "chr1\t0\t5\nchr1\t0\t5\nchr1\t3\t8\n");
    let output = run_pbz([
        "stat".into(),
        "-r".into(),
        bed.into_os_string(),
        store.into_os_string(),
        "depth".into(),
    ]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\tmean\n\
         chr1\t0\t5\t5.8\n\
         chr1\t0\t5\t5.8\n\
         chr1\t3\t8\t2.8\n"
    );
}

#[test]
fn median_min_max_report_integers() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let bed = write_bed(dir.path(), "chr1\t0\t5\nchr1\t3\t8\n");
    for (stat, expected) in [
        ("median", "chr1\t0\t5\t5\nchr1\t3\t8\t0\n"),
        ("min", "chr1\t0\t5\t5\nchr1\t3\t8\t0\n"),
        ("max", "chr1\t0\t5\t7\nchr1\t3\t8\t7\n"),
    ] {
        let output = run_pbz([
            "stat".into(),
            "--no-header".into(),
            "-s".into(),
            stat.into(),
            "-r".into(),
            bed.as_os_str().to_owned(),
            store.as_os_str().to_owned(),
            "depth".into(),
        ]);
        assert_eq!(stdout_of(&output), expected, "stat {stat}");
    }
}

#[test]
fn hist_aggregates_whole_genome() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz([
        "stat".into(),
        "-s".into(),
        "hist".into(),
        store.into_os_string(),
        "depth".into(),
    ]);
    assert_eq!(stdout_of(&output), "#value\tcount\n0\t11\n5\t3\n7\t2\n");
}

#[test]
fn hist_counts_union_of_overlapping_regions_once() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let bed = write_bed(dir.path(), "chr1\t0\t5\nchr1\t3\t8\n");
    let output = run_pbz([
        "stat".into(),
        "-s".into(),
        "hist".into(),
        "-r".into(),
        bed.into_os_string(),
        store.into_os_string(),
        "depth".into(),
    ]);
    // union 0..8 -> 5,5,5,7,7,0,0,0; a multiset over both rows would count 10
    assert_eq!(stdout_of(&output), "#value\tcount\n0\t3\n5\t3\n7\t2\n");
}

#[test]
fn column_subset_orders_output_by_flag() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz([
        "stat".into(),
        "-c".into(),
        "s2".into(),
        store.into_os_string(),
        "af".into(),
    ]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts2\n\
         chr1\t0\t10\t1\n\
         chr2\t0\t6\tNaN\n"
    );
}

#[test]
fn median_on_float_track_fails_with_clear_error() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let output = run_pbz([
        "stat".into(),
        "-s".into(),
        "median".into(),
        store.into_os_string(),
        "af".into(),
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("median"), "{stderr}");
    assert!(stderr.contains("f32"), "{stderr}");
}

#[test]
fn bgzip_bed_input_matches_plain() {
    use std::io::Write as _;
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let plain = write_bed(dir.path(), "chr1\t0\t5\n");
    let gz = dir.path().join("regions.bed.gz");
    let mut writer = noodles_bgzf::io::Writer::new(std::fs::File::create(&gz).unwrap());
    writer.write_all(b"chr1\t0\t5\n").unwrap();
    writer.finish().unwrap();
    let expected = stdout_of(&run_pbz([
        "stat".into(),
        "-r".into(),
        plain.into_os_string(),
        store.as_os_str().to_owned(),
        "depth".into(),
    ]));
    let actual = stdout_of(&run_pbz([
        "stat".into(),
        "-r".into(),
        gz.into_os_string(),
        store.into_os_string(),
        "depth".into(),
    ]));
    assert_eq!(actual, expected);
}
