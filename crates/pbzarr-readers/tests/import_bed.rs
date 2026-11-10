//! End-to-end: synthesize a bgzipped+tabix BED, import one selected column via
//! `from_bed_matrix`, read back via `Dataset::read_region`.

mod common;

use pbzarr::dataset::Dataset;
use pbzarr::genome::{Contig, Genome};
use pbzarr::import::{Config, Source};
use pbzarr_readers::{BedImportOptions, from_bed_matrix};
use tempfile::TempDir;

use common::write_bed_bgzip_tabix;

#[test]
fn single_column_selection_imports_a_scalar_track() {
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
    let store = dir.path().join("out.pbz");
    from_bed_matrix(
        store.as_path(),
        &[Source::labeled(bed, "s1")],
        Genome::new(vec![Contig {
            name: "chr1".into(),
            length: 60,
        }])
        .unwrap(),
        &BedImportOptions {
            fields: Some(vec!["coverage".into()]),
            ..BedImportOptions::default()
        },
        Config::default(),
    )
    .unwrap();

    let ds = Dataset::open(&store).unwrap();
    assert_eq!(ds.variables()["coverage"].dims, ["position"]);
    // Exhaustive inference min-widths 4..9 to uint8.
    let a = ds
        .read_region::<u8>("coverage", &"chr1".parse().unwrap())
        .unwrap();
    assert_eq!(a.shape(), &[60]);
    assert!(a.iter().take(30).all(|&v| v == 4));
    assert!(a.iter().skip(30).all(|&v| v == 9));
}
