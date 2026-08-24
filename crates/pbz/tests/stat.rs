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
