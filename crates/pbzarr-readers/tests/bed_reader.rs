//! POC smoke tests for `pbzarr_readers::BedReader`. Synthesizes a bgzipped,
//! tabix-indexed BED via htslib (skipped if unavailable).

mod common;

use ndarray::Array2;
use pbzarr::genome::{Contig, Genome};
use pbzarr::io::ValueReader;
use pbzarr_readers::BedReader;
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
fn reads_int_column_expanding_runs() {
    if !htslib_available() {
        eprintln!("skip bed_reader::reads_int_column_expanding_runs: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    // header: chrom start end coverage score  -> value column "coverage" is index 3.
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage", "score"],
        &[
            ("chr1", 0, 10, vec!["1", "-1.0"]),
            ("chr1", 10, 20, vec!["2", "-1.0"]),
            ("chr1", 20, 50, vec!["3", "-1.0"]),
        ],
    );

    let reader: BedReader<i32> = BedReader::open(&bed, 3, genome(&[("chr1", 50)])).unwrap();
    assert_eq!(reader.n_fields(), 1);

    let mut buf: Array2<i32> = Array2::zeros((25, 1)); // read [5, 30)
    reader.read_into("chr1", 5, 30, buf.view_mut()).unwrap();

    // [5,10)->1 (5 rows), [10,20)->2 (10 rows), [20,30)->3 (10 rows)
    let col: Vec<i32> = buf.column(0).to_vec();
    assert_eq!(&col[0..5], &[1; 5]);
    assert_eq!(&col[5..15], &[2; 10]);
    assert_eq!(&col[15..25], &[3; 10]);
}

#[test]
fn uncovered_positions_stay_fill() {
    if !htslib_available() {
        eprintln!("skip bed_reader::uncovered_positions_stay_fill: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage"],
        &[("chr1", 10, 20, vec!["7"])], // only [10,20) covered
    );
    let reader: BedReader<i32> = BedReader::open(&bed, 3, genome(&[("chr1", 40)])).unwrap();

    let mut buf: Array2<i32> = Array2::from_elem((40, 1), -99); // pre-fill sentinel
    reader.read_into("chr1", 0, 40, buf.view_mut()).unwrap();

    let col: Vec<i32> = buf.column(0).to_vec();
    assert!(
        col[0..10].iter().all(|&v| v == -99),
        "gap before run untouched"
    );
    assert!(col[10..20].iter().all(|&v| v == 7), "run written");
    assert!(
        col[20..40].iter().all(|&v| v == -99),
        "gap after run untouched"
    );
}

#[test]
fn reads_float_and_bool_columns() {
    if !htslib_available() {
        eprintln!("skip bed_reader::reads_float_and_bool_columns: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "score", "is_max"],
        &[
            ("chr1", 0, 5, vec!["1.5", "true"]),
            ("chr1", 5, 10, vec!["-2.5", "false"]),
        ],
    );

    let fr: BedReader<f32> = BedReader::open(&bed, 3, genome(&[("chr1", 10)])).unwrap();
    let mut fbuf: Array2<f32> = Array2::zeros((10, 1));
    fr.read_into("chr1", 0, 10, fbuf.view_mut()).unwrap();
    assert!(fbuf.column(0).iter().take(5).all(|&v| v == 1.5));
    assert!(fbuf.column(0).iter().skip(5).all(|&v| v == -2.5));

    let br: BedReader<bool> = BedReader::open(&bed, 4, genome(&[("chr1", 10)])).unwrap();
    let mut bbuf: Array2<bool> = Array2::from_elem((10, 1), false);
    br.read_into("chr1", 0, 10, bbuf.view_mut()).unwrap();
    assert!(bbuf.column(0).iter().take(5).all(|&v| v));
    assert!(bbuf.column(0).iter().skip(5).all(|&v| !v));
}

#[test]
fn column_index_by_name_reads_header() {
    if !htslib_available() {
        eprintln!("skip bed_reader::column_index_by_name_reads_header: bgzip/tabix not on PATH");
        return;
    }
    let dir = TempDir::new().unwrap();
    let bed = write_bed_bgzip_tabix(
        dir.path(),
        "a",
        &["chrom", "start", "end", "coverage", "fire_coverage"],
        &[("chr1", 0, 10, vec!["1", "0"])],
    );
    assert_eq!(
        pbzarr_readers::column_index_by_name(&bed, "fire_coverage").unwrap(),
        4
    );
}
