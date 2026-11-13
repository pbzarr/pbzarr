//! `pbz import --zip` writes a single-file store that view and stat read.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

fn fixture(name: &str) -> OsString {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/d4")
        .join(name)
        .into_os_string()
}

fn labeled(name: &str, label: &str) -> OsString {
    let mut source = fixture(name);
    source.push(format!(":{label}"));
    source
}

fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

fn stdout_of(output: Output) -> String {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn stderr_of_failure(output: Output) -> String {
    assert!(!output.status.success(), "the command did not fail");
    String::from_utf8(output.stderr).unwrap()
}

fn import_d4(output: &Path, flags: &[&str], sources: Vec<OsString>) -> Output {
    let mut args: Vec<OsString> = vec![
        "import".into(),
        "d4".into(),
        output.into(),
        "--track".into(),
        "depth".into(),
        "--no-progress".into(),
    ];
    args.extend(flags.iter().map(OsString::from));
    args.extend(sources);
    run_pbz(args)
}

fn cohort() -> Vec<OsString> {
    vec![labeled("cohort_s0.d4", "s0"), labeled("cohort_s1.d4", "s1")]
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect()
}

#[test]
fn view_and_stat_read_a_zip_store_as_they_read_a_directory_store() {
    let dir = TempDir::new().unwrap();
    let directory = dir.path().join("cohort.pbz");
    let zip = dir.path().join("cohort.pbz.zip");
    let flagged = dir.path().join("flagged.pbz");
    stdout_of(import_d4(&directory, &[], cohort()));
    stdout_of(import_d4(&zip, &[], cohort()));
    stdout_of(import_d4(&flagged, &["--zip"], cohort()));
    assert!(zip.is_file() && flagged.is_file());

    for args in [&["view"][..], &["stat", "-s", "median"]] {
        let read = |store: &Path| {
            let mut all: Vec<OsString> = args.iter().map(OsString::from).collect();
            all.extend([store.into(), "--track".into(), "depth".into()]);
            stdout_of(run_pbz(all))
        };
        let expected = read(&directory);
        assert_eq!(read(&zip), expected, "{args:?}");
        assert_eq!(read(&flagged), expected, "{args:?}");
    }
}

#[test]
fn a_zip_store_refuses_a_second_import_and_append() {
    let dir = TempDir::new().unwrap();
    let zip = dir.path().join("cohort.pbz");
    stdout_of(import_d4(&zip, &["--zip"], cohort()));

    let again = stderr_of_failure(import_d4(&zip, &["--zip"], cohort()));
    assert!(again.contains("A zip store is written once"), "{again}");

    let append = stderr_of_failure(run_pbz([
        "append".into(),
        "d4".into(),
        zip.clone().into(),
        "--track".into(),
        "depth".into(),
        "--no-progress".into(),
        labeled("cohort_s2.d4", "s2"),
    ]));
    assert!(append.contains("Zip stores are written once"), "{append}");
}

#[test]
fn a_failed_zip_import_leaves_no_file() {
    let dir = TempDir::new().unwrap();
    let zip = dir.path().join("cohort.pbz.zip");
    let sources = vec![labeled("cohort_s0.d4", "s0"), labeled("cohort_s1.d4", "s0")];
    let failed = stderr_of_failure(import_d4(&zip, &[], sources));
    assert!(failed.contains("duplicate labels"), "{failed}");
    assert!(entries(dir.path()).is_empty(), "{:?}", entries(dir.path()));
}
