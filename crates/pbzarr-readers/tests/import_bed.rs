//! End-to-end: synthesize bgzipped+tabix BEDs, import one column via `from_bed`,
//! read back via `Track::read_region`. Skipped if bgzip/tabix are unavailable.

mod common;

use std::path::PathBuf;

use ndarray::{Ix1, Ix2};
use pbzarr::genome::{Contig, Genome};
use pbzarr::import::Config;
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{BedSource, from_bed};
use tempfile::TempDir;

use common::{htslib_available, write_bed_bgzip_tabix};

fn genome(contigs: &[(&str, u64)]) -> Genome {
    Genome::new(
        contigs
            .iter()
            .map(|(n, l)| Contig {
                name: (*n).to_owned(),
                length: *l,
            })
            .collect(),
    )
    .unwrap()
}

fn src(path: PathBuf, label: &str) -> BedSource {
    BedSource {
        path,
        column_label: Some(label.to_owned()),
    }
}

#[test]
fn single_sample_scalar_import() {
    if !htslib_available() {
        eprintln!("skip import_bed::single_sample_scalar_import: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "coverage"],
        &[("chr1", 0, 30, vec!["4"]), ("chr1", 30, 60, vec!["9"])],
    );
    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();

    from_bed::<i32>(
        &mut store,
        "coverage",
        &[src(bed, "s1")],
        3,
        genome(&[("chr1", 60)]),
        Config::default(),
    )
    .unwrap();

    let track = store.track("coverage").unwrap();
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 60,
    };
    let arr = track.read_region::<i32>(&region).unwrap();
    let a = arr.into_dimensionality::<Ix1>().unwrap();
    assert_eq!(a.shape(), &[60]);
    assert!(a.iter().take(30).all(|&v| v == 4));
    assert!(a.iter().skip(30).all(|&v| v == 9));
}

#[test]
fn multi_sample_cohort_import() {
    if !htslib_available() {
        eprintln!("skip import_bed::multi_sample_cohort_import: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let g = genome(&[("chr1", 40)]);
    let s1 = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "coverage"],
        &[("chr1", 0, 40, vec!["1"])],
    );
    let s2 = write_bed_bgzip_tabix(
        dir.path(),
        "s2",
        &["chrom", "start", "end", "coverage"],
        &[("chr1", 0, 40, vec!["2"])],
    );
    let store_path = dir.path().join("cohort.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();

    from_bed::<i32>(
        &mut store,
        "coverage",
        &[src(s1, "s1"), src(s2, "s2")],
        3,
        g,
        Config::default(),
    )
    .unwrap();

    let track = store.track("coverage").unwrap();
    assert_eq!(track.rank(), 2);
    assert_eq!(track.column_dim(), Some("sample"));
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 40,
    };
    let a = track
        .read_region::<i32>(&region)
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert_eq!(a.shape(), &[40, 2]);
    assert!(a.column(0).iter().all(|&v| v == 1));
    assert!(a.column(1).iter().all(|&v| v == 2));
}

#[test]
fn chunk_boundary_run_spans() {
    if !htslib_available() {
        eprintln!("skip import_bed::chunk_boundary_run_spans: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    // One run spans the chunk boundary at 100.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "coverage"],
        &[("chr1", 0, 250, vec!["5"])],
    );
    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();

    // force multiple physical chunks over ΣL=250
    let config = Config {
        chunk_size: Some(100),
        ..Config::default()
    };

    from_bed::<i32>(
        &mut store,
        "coverage",
        &[src(bed, "s1")],
        3,
        genome(&[("chr1", 250)]),
        config,
    )
    .unwrap();

    let track = store.track("coverage").unwrap();
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 250,
    };
    let a = track
        .read_region::<i32>(&region)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert!(
        a.iter().all(|&v| v == 5),
        "run value uniform across chunk boundaries"
    );
}
