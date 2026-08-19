//! End-to-end: synthesize a bgzipped+tabix BED, import one selected column via
//! `from_bed_matrix`, read back via `Track::read_region`. Skipped if
//! bgzip/tabix are unavailable.

mod common;

use ndarray::Ix1;
use pbzarr::genome::{Contig, Genome};
use pbzarr::import::{Config, Source};
use pbzarr::{PbzStore, Region};
use pbzarr_readers::{BedImportOptions, from_bed_matrix};
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

#[test]
fn single_column_selection_imports_a_scalar_track() {
    if !htslib_available() {
        eprintln!(
            "skip import_bed::single_column_selection_imports_a_scalar_track: bgzip/tabix not on PATH"
        );
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "s1",
        &["chrom", "start", "end", "coverage", "score"],
        &[
            ("chr1", 0, 30, vec!["4", "1.5"]),
            ("chr1", 30, 60, vec!["9", "2.5"]),
        ],
    );
    let store_path = dir.path().join("out.pbz");
    let mut store = PbzStore::create(&store_path).unwrap();

    from_bed_matrix(
        &mut store,
        &[Source::labeled(bed, "s1")],
        genome(&[("chr1", 60)]),
        &BedImportOptions {
            fields: Some(vec!["coverage".into()]),
            ..BedImportOptions::default()
        },
        Config::default(),
    )
    .unwrap();

    let track = store.track("coverage").unwrap();
    assert_eq!(track.rank(), 1);
    let region = Region {
        contig: store.genome_for("coverage").unwrap().id("chr1").unwrap(),
        start: 0,
        end: 60,
    };
    // Exhaustive inference min-widths 4..9 to uint8.
    let a = track
        .read_region::<u8>(&region)
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert_eq!(a.shape(), &[60]);
    assert!(a.iter().take(30).all(|&v| v == 4));
    assert!(a.iter().skip(30).all(|&v| v == 9));
}
