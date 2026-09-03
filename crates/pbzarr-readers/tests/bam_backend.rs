mod bam_common;

use pbzarr_readers::bam::RecordFields;
use pbzarr_readers::bam::backend::Backend;

#[test]
fn open_reads_header_genome_and_fetch_visits_records() {
    let fx = bam_common::fixture();
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

    // A zero-width window is empty, not an error.
    assert!(!b.has_records("ref1", 10, 10).unwrap());
    let mut n = 0;
    b.fetch("ref1", 10, 10, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 0);
}

#[test]
fn misnamed_cram_as_bam_extension_still_opens_as_cram() {
    let fx = bam_common::fixture();
    let dir = tempfile::TempDir::new().unwrap();

    // Content sniff (F7) must win over the misleading `.bam` extension: copy
    // the CRAM bytes (and its `.crai` index, same naming convention) to a
    // `.bam`-named path and confirm it still opens and fetches as CRAM.
    let misnamed_cram = dir.path().join("actually_cram.bam");
    std::fs::copy(&fx.cram, &misnamed_cram).unwrap();
    std::fs::copy(
        format!("{}.crai", fx.cram.display()),
        format!("{}.crai", misnamed_cram.display()),
    )
    .unwrap();

    let (mut b, genome) =
        Backend::open(&misnamed_cram, Some(&fx.fasta), RecordFields::Full).unwrap();
    assert_eq!(genome.contigs().len(), 2);
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);

    // A zero-width window is empty, not an error, on the CRAM arm too.
    assert!(!b.has_records("ref1", 10, 10).unwrap());
    let mut n = 0;
    b.fetch("ref1", 10, 10, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 0);

    // Same sniff, opposite direction: bgzf magic on a `.cram`-named path
    // must route into the BAM arm, which ignores `reference` entirely.
    let misnamed_bam = dir.path().join("actually_bam.cram");
    std::fs::copy(&fx.bam, &misnamed_bam).unwrap();
    std::fs::copy(
        format!("{}.bai", fx.bam.display()),
        format!("{}.bai", misnamed_bam.display()),
    )
    .unwrap();

    let (mut b, genome) = Backend::open(&misnamed_bam, None, RecordFields::Full).unwrap();
    assert_eq!(genome.contigs().len(), 2);
    let mut n = 0;
    b.fetch("ref1", 0, 100, &mut |_| {
        n += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 9);

    let bogus_reference = dir.path().join("does-not-exist.fasta");
    // This is the path the drop-order CRAM UAF used to expose: `set_reference`
    // fails and `reader` drops early. The fork pinned in Cargo.toml destroys
    // the index before `hts_close`, so this must error cleanly, not crash.
    assert!(Backend::open(&fx.cram, Some(&bogus_reference), RecordFields::Full).is_err());
}
