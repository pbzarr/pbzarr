//! End-to-end: synthesize a small d4 file via the system `d4tools` (skipped
//! if unavailable), import via `import_d4`, read back via `Track::read_region`.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use pbzarr::ingest::{D4Source, ImportConfig, import_d4};
use pbzarr::io::Dtype;
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};
use tempfile::TempDir;

fn have_d4tools() -> bool {
    Command::new("d4tools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Write a tiny bedGraph-based d4 where positions [i*10, (i+1)*10) carry value
/// `(i % 50) + 1` (1-indexed bands of 10), then call `d4tools create --genome`.
fn write_synthetic_d4(tmp: &Path, chrom: &str, len: u32) -> std::path::PathBuf {
    let sizes_path = tmp.join("genome.sizes");
    let mut sf = std::fs::File::create(&sizes_path).unwrap();
    writeln!(sf, "{chrom}\t{len}").unwrap();
    drop(sf);

    let bg_path = tmp.join("data.bedgraph");
    let mut bf = std::fs::File::create(&bg_path).unwrap();
    for i in 0..(len / 10) {
        let s = i * 10;
        let e = (i + 1) * 10;
        let v = (i % 50) + 1;
        writeln!(bf, "{chrom}\t{s}\t{e}\t{v}").unwrap();
    }
    drop(bf);

    let d4_path = tmp.join("out.d4");
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

#[test]
fn import_one_d4_into_scalar_track() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let d4 = write_synthetic_d4(dir.path(), "chr1", 1_000);

    let store_path = dir.path().join("out.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 1_000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&store_path, genome, None).unwrap();
    store
        .create_track("depth", TrackConfig::scalar(Dtype::U32))
        .unwrap();

    import_d4(
        &mut store,
        "depth",
        &[D4Source {
            path: d4,
            sample_label: None,
        }],
        ImportConfig::default(),
    )
    .unwrap();

    let region = Region {
        contig: store.genome().id("chr1").unwrap(),
        start: 0,
        end: 1_000,
    };
    let got = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    let arr = got.into_dimensionality::<ndarray::Ix1>().unwrap();

    for i in 0..100u32 {
        let v = (i % 50) + 1;
        for p in (i * 10)..((i + 1) * 10) {
            assert_eq!(arr[p as usize], v, "pos {p}: expected {v}");
        }
    }
}
