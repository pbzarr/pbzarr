mod bam_common;
use std::path::PathBuf;

use pbzarr_readers::bam::RecordFields;
use pbzarr_readers::bam::backend::Backend;

#[test]
fn open_reads_header_genome_and_fetch_visits_records() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    let (mut b, genome) = Backend::open(&fx.bam, None, RecordFields::Full).unwrap();
    assert_eq!(genome.contigs().len(), 2);
    assert_eq!(genome.get(genome.id("ref1").unwrap()).unwrap().length, 100);
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |r| {
        assert!(!r.cigar.is_empty());
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);
    assert!(b.has_records("ref1", 0, 100).unwrap());
    assert!(!b.has_records("ref2", 0, 60).unwrap());
}

#[test]
fn cram_backend_matches_bam() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    let (mut b, _) = Backend::open(&fx.cram, Some(&fx.fasta), RecordFields::Full).unwrap();
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);
}

#[test]
fn misnamed_cram_as_bam_extension_still_opens_as_cram() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    // Content sniff (F7) must win over the misleading `.bam` extension: copy
    // the CRAM bytes (and its `.crai` index, same naming convention) to a
    // `.bam`-named path and confirm it still opens and fetches as CRAM.
    let misnamed = dir.path().join("actually_cram.bam");
    std::fs::copy(&fx.cram, &misnamed).unwrap();
    let cram_index = PathBuf::from(format!("{}.crai", fx.cram.display()));
    std::fs::copy(&cram_index, format!("{}.crai", misnamed.display())).unwrap();

    let (mut b, genome) = Backend::open(&misnamed, Some(&fx.fasta), RecordFields::Full).unwrap();
    assert_eq!(genome.contigs().len(), 2);
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);
}

#[test]
fn misnamed_bam_as_cram_extension_still_opens_as_bam() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    // Same sniff, opposite direction: bgzf magic on a `.cram`-named path
    // must route into the BAM arm, which ignores `reference` entirely.
    let misnamed = dir.path().join("actually_bam.cram");
    std::fs::copy(&fx.bam, &misnamed).unwrap();
    let bam_index = PathBuf::from(format!("{}.bai", fx.bam.display()));
    std::fs::copy(&bam_index, format!("{}.bai", misnamed.display())).unwrap();

    let (mut b, genome) = Backend::open(&misnamed, None, RecordFields::Full).unwrap();
    assert_eq!(genome.contigs().len(), 2);
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);
}

#[test]
fn zero_width_window_is_empty_not_an_error() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };

    let (mut bam, _) = Backend::open(&fx.bam, None, RecordFields::Full).unwrap();
    assert!(!bam.has_records("ref1", 10, 10).unwrap());
    let mut n = 0;
    bam.fetch("ref1", 10, 10, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 0);

    let (mut cram, _) = Backend::open(&fx.cram, Some(&fx.fasta), RecordFields::Full).unwrap();
    assert!(!cram.has_records("ref1", 10, 10).unwrap());
    let mut n = 0;
    cram.fetch("ref1", 10, 10, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 0);
}

#[test]
fn cram_open_with_bogus_reference_errors_without_crashing() {
    let dir = tempfile::TempDir::new().unwrap();
    let Some(fx) = bam_common::fixture(dir.path()) else {
        return;
    };
    let bogus_reference = dir.path().join("does-not-exist.fasta");
    // This is the path the drop-order CRAM UAF used to expose: `set_reference`
    // fails and `reader` drops early. The fork pinned in Cargo.toml destroys
    // the index before `hts_close`, so this must error cleanly, not crash.
    assert!(Backend::open(&fx.cram, Some(&bogus_reference), RecordFields::Full).is_err());
}
