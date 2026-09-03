use std::ffi::OsString;
use std::path::Path;
use std::process::{Command, Output};

use tempfile::TempDir;

fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn dry_run_prints_plan_and_estimate_without_creating_the_store() {
    let bam = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/bam/reads.bam");
    let dir = TempDir::new().unwrap();
    let output = dir.path().join("depth.pbz");

    let command = run_pbz([
        "import".into(),
        "bam".into(),
        "-o".into(),
        output.as_os_str().to_owned(),
        "--track".into(),
        "cov".into(),
        "--mode".into(),
        "composition".into(),
        "--dry-run".into(),
        bam.into_os_string(),
    ]);
    assert!(
        command.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&command.stderr)
    );
    let stdout = String::from_utf8_lossy(&command.stdout);
    assert!(stdout.contains("track cov "), "{stdout}");
    assert!(stdout.contains("cov_ins"), "{stdout}");
    assert!(stdout.contains("tasks/worker"), "{stdout}");
    assert!(!output.exists(), "dry run created the output store");
}
