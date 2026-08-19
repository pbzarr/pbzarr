use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

/// True if `samtools` is on PATH. With `PBZ_REQUIRE_TOOLS` set (CI), a
/// missing tool panics instead of letting the test self-skip.
fn have_samtools() -> bool {
    let ok = Command::new("samtools")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    if !ok && std::env::var_os("PBZ_REQUIRE_TOOLS").is_some() {
        panic!("PBZ_REQUIRE_TOOLS is set but samtools is not on PATH");
    }
    ok
}

/// One sorted, indexed single-read BAM; a dry run only reads its header.
fn write_bam(dir: &Path) -> PathBuf {
    let sam = dir.join("reads.sam");
    std::fs::write(
        &sam,
        "@HD\tVN:1.6\tSO:unsorted\n\
         @SQ\tSN:chr1\tLN:100\n\
         @SQ\tSN:chr2\tLN:60\n\
         r1\t0\tchr1\t11\t60\t20M\t*\t0\t0\tAAAAAAAAAAAAAAAAAAAA\t????????????????????\n",
    )
    .unwrap();
    let bam = dir.join("reads.bam");
    assert!(
        Command::new("samtools")
            .args(["sort", "-o"])
            .arg(&bam)
            .arg(&sam)
            .status()
            .unwrap()
            .success(),
        "samtools sort failed"
    );
    assert!(
        Command::new("samtools")
            .args(["index"])
            .arg(&bam)
            .status()
            .unwrap()
            .success(),
        "samtools index failed"
    );
    bam
}

fn run_pbz(args: impl IntoIterator<Item = OsString>) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pbz"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn dry_run_prints_plan_and_estimate_without_creating_the_store() {
    if !have_samtools() {
        eprintln!("skip: samtools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let bam = write_bam(dir.path());
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
