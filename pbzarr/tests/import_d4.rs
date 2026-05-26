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
        .create_track("depth", TrackConfig::new(Dtype::U32))
        .unwrap();

    import_d4(
        &store,
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

/// Regression: d4 file orders contigs differently from the pbz store. The
/// pipeline must resolve `ContigId` in the *store's* genome to a name, then
/// hand the name to the reader (which looks it up in *its* genome). Earlier
/// code shared the `ContigId` directly and silently wrote chr2 data into
/// chr1's array when ordering differed.
#[test]
fn import_d4_multi_contig_out_of_order_writes_correct_contigs() {
    if !have_d4tools() {
        eprintln!("skip: d4tools not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();

    // d4 has chr2 first, chr1 second (reverse of pbz order).
    let d4_path = dir.path().join("a.d4");
    let bg_path = d4_path.with_extension("bedgraph");
    let sizes_path = d4_path.with_extension("sizes");
    // chr2 values: 100. chr1 values: 7.
    std::fs::write(&bg_path, "chr2\t0\t500\t100\nchr1\t0\t300\t7\n").unwrap();
    std::fs::write(&sizes_path, "chr2\t500\nchr1\t300\n").unwrap();
    let status = Command::new("d4tools")
        .args(["create", "--genome"])
        .arg(&sizes_path)
        .arg(&bg_path)
        .arg(&d4_path)
        .status()
        .unwrap();
    assert!(status.success(), "d4tools create failed");

    // pbz store has chr1 first, chr2 second.
    let store_path = dir.path().join("out.pbz");
    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 300,
        },
        Contig {
            name: "chr2".into(),
            length: 500,
        },
    ])
    .unwrap();
    let mut store = PbzStore::create(&store_path, genome, None).unwrap();
    store
        .create_track("depth", TrackConfig::new(Dtype::U32))
        .unwrap();

    import_d4(
        &store,
        "depth",
        &[D4Source {
            path: d4_path,
            sample_label: None,
        }],
        ImportConfig::default(),
    )
    .unwrap();

    let chr1_id = store.genome().id("chr1").unwrap();
    let chr2_id = store.genome().id("chr2").unwrap();
    let chr1 = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&Region {
            contig: chr1_id,
            start: 0,
            end: 300,
        })
        .unwrap()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let chr2 = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&Region {
            contig: chr2_id,
            start: 0,
            end: 500,
        })
        .unwrap()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    assert!(chr1.iter().all(|&v| v == 7), "chr1 should have value 7");
    assert!(chr2.iter().all(|&v| v == 100), "chr2 should have value 100");
}
