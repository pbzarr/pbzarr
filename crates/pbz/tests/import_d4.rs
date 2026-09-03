//! End-to-end: import the committed d4 fixture under `fixtures/d4/` via
//! `pbz import d4`, then read it back with `pbz stat`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name)
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

fn labeled(path: &Path, label: &str) -> OsString {
    format!("{}:{label}", path.display()).into()
}

#[test]
fn import_d4_then_stat_mean() {
    let dir = TempDir::new().unwrap();
    // chr1 length 1000, bands of 10, band i valued (i % 50) + 1.
    let d4 = fixture("banded_1k.d4");

    let store = dir.path().join("store.pbz");
    let import = run_pbz([
        "import".into(),
        "d4".into(),
        "-o".into(),
        store.as_os_str().to_owned(),
        "--track".into(),
        "depth".into(),
        d4.as_os_str().to_owned(),
        "--no-progress".into(),
    ]);
    assert!(
        import.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&import.stderr)
    );

    let mean = run_pbz(["stat".into(), store.into_os_string(), "depth".into()]);
    assert_eq!(
        stdout_of(&mean),
        "#chrom\tstart\tend\tmean\nchr1\t0\t1000\t25.5\n"
    );

    // Two labeled sources sharing the fixture form a 2-column cohort track.
    let cohort = dir.path().join("cohort.pbz");
    let cohort_import = run_pbz([
        "import".into(),
        "d4".into(),
        "-o".into(),
        cohort.as_os_str().to_owned(),
        "--track".into(),
        "depth".into(),
        "--no-progress".into(),
        labeled(&d4, "s1"),
        labeled(&d4, "s2"),
    ]);
    assert!(
        cohort_import.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&cohort_import.stderr)
    );

    // Values 1..=50 each cover 20 positions; the lower-middle rule (index
    // (1000-1)/2 = 499, 0-based) picks 25 for both identical columns.
    let median = run_pbz([
        "stat".into(),
        "-s".into(),
        "median".into(),
        cohort.into_os_string(),
        "depth".into(),
    ]);
    assert_eq!(
        stdout_of(&median),
        "#chrom\tstart\tend\ts1\ts2\nchr1\t0\t1000\t25\t25\n"
    );
}
