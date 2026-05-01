mod fixtures;

use assert_cmd::Command;
use fixtures::{build_bgz_bed, build_chrom_sizes};
use tempfile::TempDir;

fn pbz() -> Command {
    Command::cargo_bin("pbz").unwrap()
}

#[test]
fn import_single_bed3_creates_1d_bool_track() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\nchr1\t300\t400\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000), ("chr2", 500_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    pbz()
        .args(["info", store_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("track: callable"))
        .stdout(predicates::str::contains("dtype: bool"));
}

#[test]
fn import_two_bed3_with_sample_suffix_creates_2d_track() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let a = build_bgz_bed("chr1\t100\t200\n");
    let b = build_bgz_bed("chr1\t150\t250\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            &format!("{}:A", a.path().display()),
            "-i",
            &format!("{}:B", b.path().display()),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    pbz()
        .args([
            "list",
            store_path.to_str().unwrap(),
            "--samples",
            "callable",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("A"))
        .stdout(predicates::str::contains("B"));
}

#[test]
fn import_single_bedgraph_default_f32() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\t1.5\nchr1\t300\t400\t2.5\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "score",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    pbz()
        .args(["info", store_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("dtype: float32"));
}

#[test]
fn import_two_bedgraph_with_dtype_uint16() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let a = build_bgz_bed("chr1\t100\t200\t10\n");
    let b = build_bgz_bed("chr1\t150\t250\t20\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "depth",
            "--dtype",
            "uint16",
            "-i",
            &format!("{}:A", a.path().display()),
            "-i",
            &format!("{}:B", b.path().display()),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    pbz()
        .args(["info", store_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicates::str::contains("dtype: uint16"));
}

#[test]
fn second_import_into_existing_store_rejects_contigs_flag() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "t1",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    let bed2 = build_bgz_bed("chr1\t300\t400\n");
    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "t2",
            "-i",
            bed2.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--contigs is rejected"));
}

#[test]
fn import_overlapping_intervals_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t10\t100\nchr1\t50\t200\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("overlapping"));
}

#[test]
fn import_unsorted_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\nchr1\t10\t20\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("must be sorted"));
}

#[test]
fn import_unknown_contig_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chrUnknown\t10\t20\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not present"));
}

#[test]
fn import_past_contig_end_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t999000\t1500000\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("past contig end"));
}

#[test]
fn import_zero_length_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t100\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("zero-length"));
}

#[test]
fn import_dtype_with_mask_rejected() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "callable",
            "--dtype",
            "uint8",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("only valid for bedGraph"));
}

#[test]
fn import_value_out_of_range_for_uint8() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\t300\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "depth",
            "--dtype",
            "uint8",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("uint8"));
}

#[test]
fn import_mixed_flavor_in_one_file_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\nchr1\t300\t400\t1.5\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "t",
            "-i",
            bed.path().to_str().unwrap(),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("flavor"));
}

#[test]
fn import_mixed_flavor_across_files_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let mask = build_bgz_bed("chr1\t100\t200\n");
    let bg = build_bgz_bed("chr1\t300\t400\t1.5\n");
    let contigs = build_chrom_sizes(&[("chr1", 1_000_000)]);

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "t",
            "-i",
            &format!("{}:A", mask.path().display()),
            "-i",
            &format!("{}:B", bg.path().display()),
            "--contigs",
            contigs.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("flavor"));
}

#[test]
fn import_missing_contigs_for_new_store_errors() {
    let store_dir = TempDir::new().unwrap();
    let store_path = store_dir.path().join("s.pbz.zarr");
    let bed = build_bgz_bed("chr1\t100\t200\n");

    pbz()
        .args([
            "import",
            store_path.to_str().unwrap(),
            "--track",
            "t",
            "-i",
            bed.path().to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--contigs"));
}
