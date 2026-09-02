//! End-to-end: synthesize a small d4 file via the system `d4tools` (skipped
//! if unavailable), import it via `pbz import d4`, then read it back with
//! `pbz stat`.

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

/// True if `d4tools` is on PATH. With `PBZ_REQUIRE_TOOLS` set (CI), a
/// missing tool panics instead of letting the test self-skip.
fn have_d4tools() -> bool {
    let ok = Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    if !ok && std::env::var_os("PBZ_REQUIRE_TOOLS").is_some() {
        panic!("PBZ_REQUIRE_TOOLS is set but d4tools is not on PATH");
    }
    ok
}

/// Write a single-contig d4 file by piping a bedGraph through `d4tools
/// create`. Each `values[i]` covers position `i`.
fn write_d4(dir: &Path, name: &str, chrom: &str, values: &[i32]) -> PathBuf {
    let sizes_path = dir.join(format!("{name}.sizes"));
    std::fs::write(&sizes_path, format!("{chrom}\t{}\n", values.len())).unwrap();

    let bg_path = dir.join(format!("{name}.bedgraph"));
    let mut bg = std::fs::File::create(&bg_path).unwrap();
    for (pos, value) in values.iter().enumerate() {
        writeln!(bg, "{chrom}\t{pos}\t{}\t{value}", pos + 1).unwrap();
    }
    drop(bg);

    let d4_path = dir.join(format!("{name}.d4"));
    let status = Command::new("d4tools")
        .args(["create", "--genome"])
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");
    d4_path
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
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let fixture = write_d4(
        dir.path(),
        "fixture",
        "chr1",
        &[5, 5, 5, 7, 7, 0, 0, 0, 0, 0],
    );

    let store = dir.path().join("store.pbz");
    let import = run_pbz([
        "import".into(),
        "d4".into(),
        "-o".into(),
        store.as_os_str().to_owned(),
        "--track".into(),
        "depth".into(),
        fixture.as_os_str().to_owned(),
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
        "#chrom\tstart\tend\tmean\nchr1\t0\t10\t2.9\n"
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
        labeled(&fixture, "s1"),
        labeled(&fixture, "s2"),
    ]);
    assert!(
        cohort_import.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&cohort_import.stderr)
    );

    // Sorted values are 0,0,0,0,0,5,5,5,7,7; the lower-middle rule (index
    // (10-1)/2 = 4, 0-based) picks 0 for both identical columns.
    let median = run_pbz([
        "stat".into(),
        "-s".into(),
        "median".into(),
        cohort.into_os_string(),
        "depth".into(),
    ]);
    assert_eq!(
        stdout_of(&median),
        "#chrom\tstart\tend\ts1\ts2\nchr1\t0\t10\t0\t0\n"
    );
}
