use pbzarr::{Contig, Genome, PbzStore};
use tempfile::TempDir;

#[test]
fn create_writes_contigs_and_root_attrs() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 1_000_000 },
        Contig { name: "chr2".into(), length: 500_000 },
    ])
    .unwrap();

    let _store = PbzStore::create(&path, genome, Some("GRCh38".into())).unwrap();

    // Zarr v3 store on disk
    assert!(path.join("zarr.json").exists());
    assert!(path.join("contigs").join("zarr.json").exists());
    assert!(path.join("contig_lengths").join("zarr.json").exists());
}
