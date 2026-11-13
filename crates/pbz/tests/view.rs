use std::fs::File;
use std::io::Read;

use noodles_bgzf as bgzf;
use tempfile::TempDir;

mod common;
use common::{add_rank_three_variable, build_null_store, build_store, run_pbz, stdout_of};

#[test]
fn integer_track_streams_collapsed_rows_in_genome_order() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());

    let ambiguous = run_pbz(["view".into(), store.as_os_str().to_owned()]);
    assert!(!ambiguous.status.success());
    let error = String::from_utf8_lossy(&ambiguous.stderr);
    assert!(error.contains("store has 2 tracks"), "{error}");

    let output = run_pbz([
        "view".into(),
        store.into_os_string(),
        "--track".into(),
        "depth".into(),
    ]);
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
    let output = run_pbz([
        "view".into(),
        store.into_os_string(),
        "--track".into(),
        "af".into(),
    ]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t2\t4\t0.5\t1\n\
         chr1\t4\t6\t0.25\t1\n"
    );
}

#[test]
fn integer_null_is_missing_in_view_and_stat() {
    let dir = TempDir::new().unwrap();
    let store = build_null_store(dir.path());
    let view = run_pbz(["view".into(), store.as_os_str().to_owned()]);
    assert_eq!(
        stdout_of(&view),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t2\t4\t4\t2\n\
         chr1\t4\t5\tNaN\t6\n"
    );
    let stat = run_pbz([
        "stat".into(),
        store.into_os_string(),
        "-r".into(),
        "chr1".into(),
    ]);
    assert_eq!(
        stdout_of(&stat),
        "#chrom\tstart\tend\ts1\ts2\n\
         chr1\t0\t10\t4\t3.33333\n"
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
        "--track".into(),
        "af".into(),
    ]);
    assert_eq!(stdout_of(&output), "chr1\t2\t6\t1\n");

    let output = run_pbz([
        "view".into(),
        dir.path().join("store.pbz").into_os_string(),
        "--track".into(),
        "af".into(),
        "-c".into(),
        "s2,s1".into(),
    ]);
    assert_eq!(
        stdout_of(&output),
        "#chrom\tstart\tend\ts2\ts1\n\
         chr1\t2\t4\t1\t0.5\n\
         chr1\t4\t6\t1\t0.25\n"
    );
}

#[test]
fn regions_clip_rows_and_refuse_ends_past_the_contig() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let view = |region: &str| {
        run_pbz([
            "view".into(),
            store.as_os_str().to_owned(),
            "--track".into(),
            "depth".into(),
            "-r".into(),
            region.into(),
        ])
    };
    assert_eq!(
        stdout_of(&view("chr1:4-8")),
        "#chrom\tstart\tend\tdepth\n\
         chr1\t4\t5\t7\n\
         chr1\t5\t8\t0\n"
    );
    let past_end = view("chr1:0-11");
    assert!(!past_end.status.success());
    let error = String::from_utf8_lossy(&past_end.stderr);
    assert!(error.contains("outside length 10"), "{error}");
}

#[test]
fn gz_output_writes_bgzf_with_identical_content() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    let gz = dir.path().join("out.bedgraph.gz");
    let plain = stdout_of(&run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "--track".into(),
        "depth".into(),
    ]));
    let output = run_pbz([
        "view".into(),
        "-o".into(),
        gz.as_os_str().to_owned(),
        store.into_os_string(),
        "--track".into(),
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
        "--track".into(),
        "depth".into(),
    ]));
    let capped = stdout_of(&run_pbz([
        "view".into(),
        "-t".into(),
        "1".into(),
        store.into_os_string(),
        "--track".into(),
        "depth".into(),
    ]));
    assert_eq!(capped, default_run);
}

#[test]
fn rank_three_variable_is_rejected_by_name_and_dimensions() {
    let dir = TempDir::new().unwrap();
    let store = build_store(dir.path());
    add_rank_three_variable(&store, "callable");

    let output = run_pbz([
        "view".into(),
        store.as_os_str().to_owned(),
        "--track".into(),
        "callable".into(),
    ]);
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("callable"), "{error}");
    assert!(error.contains("position, sample, threshold"), "{error}");

    let output = run_pbz([
        "stat".into(),
        store.into_os_string(),
        "--track".into(),
        "callable".into(),
    ]);
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("position, sample, threshold"), "{error}");
}
