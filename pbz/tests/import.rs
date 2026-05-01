mod fixtures;

use assert_cmd::Command;
use std::process;

fn d4tools_available() -> bool {
    process::Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Synthesize a tiny D4 file using `d4tools create` from a bedGraph.
/// Returns the D4 path; the caller's TempDir keeps it alive.
fn synth_d4(
    tmpdir: &std::path::Path,
    name: &str,
    contigs: &[(&str, u64)],
    value: u32,
) -> std::path::PathBuf {
    use std::io::Write;
    let sizes_path = tmpdir.join(format!("{name}.sizes"));
    let mut sf = std::fs::File::create(&sizes_path).unwrap();
    for (c, l) in contigs {
        writeln!(sf, "{c}\t{l}").unwrap();
    }
    drop(sf);
    let bg_path = tmpdir.join(format!("{name}.bedgraph"));
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for (c, l) in contigs {
        writeln!(bf, "{c}\t0\t{l}\t{value}").unwrap();
    }
    drop(bf);
    let d4_path = tmpdir.join(format!("{name}.d4"));
    let status = process::Command::new("d4tools")
        .arg("create")
        .arg("--genome")
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");
    d4_path
}

#[test]
fn import_two_d4s_round_trips() {
    if !d4tools_available() {
        eprintln!("skipping import_two_d4s_round_trips: d4tools not in PATH");
        return;
    }
    let f = fixtures::StoreFixture::empty();
    let tmpdir = f._dir.path();
    let store_path = tmpdir.join("imported.pbz.zarr");

    let d4_a = synth_d4(tmpdir, "a", &[("chr1", 1000), ("chr2", 500)], 5);
    let d4_b = synth_d4(tmpdir, "b", &[("chr1", 1000), ("chr2", 500)], 7);

    Command::cargo_bin("pbz")
        .unwrap()
        .args(["import"])
        .arg(&store_path)
        .args(["--track", "depths"])
        .args(["-i"])
        .arg(&d4_a)
        .args(["-i"])
        .arg(&d4_b)
        .args(["--chunk-size", "100"])
        .args(["--sample-chunk-size", "2"])
        .args(["--no-progress"])
        .assert()
        .success();

    // Verify with cat: chr1:0-3 should show two columns (a=5, b=7) on three rows.
    let out = Command::cargo_bin("pbz")
        .unwrap()
        .args(["cat"])
        .arg(&store_path)
        .arg("depths")
        .args(["--region", "chr1:0-3"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("contig\tpos\ta\tb"));
    assert!(s.contains("chr1\t0\t5\t7"));
    assert!(s.contains("chr1\t1\t5\t7"));
    assert!(s.contains("chr1\t2\t5\t7"));
}

#[test]
fn import_filelist_round_trips() {
    if !d4tools_available() {
        eprintln!("skipping import_filelist_round_trips: d4tools not in PATH");
        return;
    }
    let f = fixtures::StoreFixture::empty();
    let tmpdir = f._dir.path();
    let store_path = tmpdir.join("imported.pbz.zarr");

    let d4_a = synth_d4(tmpdir, "a", &[("chr1", 1000), ("chr2", 500)], 3);
    let d4_b = synth_d4(tmpdir, "b", &[("chr1", 1000), ("chr2", 500)], 9);

    let list_path = tmpdir.join("inputs.txt");
    std::fs::write(
        &list_path,
        format!(
            "# inputs\n{}:sample_a\n{}:sample_b\n",
            d4_a.display(),
            d4_b.display(),
        ),
    )
    .unwrap();

    Command::cargo_bin("pbz")
        .unwrap()
        .args(["import"])
        .arg(&store_path)
        .args(["--track", "depths"])
        .args(["-f"])
        .arg(&list_path)
        .args(["--chunk-size", "100"])
        .args(["--no-progress"])
        .assert()
        .success();

    let out = Command::cargo_bin("pbz")
        .unwrap()
        .args(["list"])
        .arg(&store_path)
        .args(["--samples", "depths"])
        .output()
        .unwrap();
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(s.contains("sample_a"));
    assert!(s.contains("sample_b"));
}

#[test]
fn import_existing_store_with_track_collision_errors() {
    if !d4tools_available() {
        eprintln!(
            "skipping import_existing_store_with_track_collision_errors: d4tools not in PATH"
        );
        return;
    }
    let f = fixtures::StoreFixture::empty();
    let tmpdir = f._dir.path();
    let store_path = tmpdir.join("imported.pbz.zarr");

    let d4_a = synth_d4(tmpdir, "a", &[("chr1", 1000), ("chr2", 500)], 1);

    Command::cargo_bin("pbz")
        .unwrap()
        .args(["import"])
        .arg(&store_path)
        .args(["--track", "depths", "-i"])
        .arg(&d4_a)
        .args(["--chunk-size", "100", "--no-progress"])
        .assert()
        .success();

    Command::cargo_bin("pbz")
        .unwrap()
        .args(["import"])
        .arg(&store_path)
        .args(["--track", "depths", "-i"])
        .arg(&d4_a)
        .args(["--chunk-size", "100", "--no-progress"])
        .assert()
        .failure();
}
