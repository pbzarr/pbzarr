use std::path::{Path, PathBuf};

use tempfile::TempDir;

mod common;
use common::{build_store, run_pbz, stdout_of};

fn write_bed(dir: &Path, text: &str) -> PathBuf {
    let path = dir.join("regions.bed");
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn whole_genome_covers_mean_hist_and_column_order() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let cases: [(&[&str], &str, &str); 4] = [
        (
            &[],
            "depth",
            "#chrom\tstart\tend\tmean\n\
             chr1\t0\t10\t2.9\n\
             chr2\t0\t6\t0\n",
        ),
        (
            &[],
            "af",
            "#chrom\tstart\tend\ts1\ts2\n\
             chr1\t0\t10\t0.375\t1\n\
             chr2\t0\t6\tNaN\tNaN\n",
        ),
        (
            &["-s", "hist"],
            "depth",
            "#value\tcount\n0\t11\n5\t3\n7\t2\n",
        ),
        (
            &["-c", "s2,s1"],
            "af",
            "#chrom\tstart\tend\ts2\ts1\n\
             chr1\t0\t10\t1\t0.375\n\
             chr2\t0\t6\tNaN\tNaN\n",
        ),
    ];
    for (flags, track, expected) in cases {
        let mut args = vec!["stat".into()];
        args.extend(flags.iter().map(|f| (*f).into()));
        args.push(store.as_os_str().to_owned());
        args.push(track.into());
        let output = run_pbz(args);
        assert_eq!(
            stdout_of(&output),
            expected,
            "flags {flags:?} track {track}"
        );
    }
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

    let float_track = run_pbz([
        "stat".into(),
        "-s".into(),
        "median".into(),
        store.into_os_string(),
        "af".into(),
    ]);
    assert!(!float_track.status.success());
    let stderr = String::from_utf8_lossy(&float_track.stderr);
    assert!(stderr.contains("median"), "{stderr}");
}
