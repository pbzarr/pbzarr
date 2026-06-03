use pbzarr::{Contig, Genome, PbzStore};
use tempfile::TempDir;

#[test]
fn create_writes_contigs_and_root_attrs() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 1_000_000,
        },
        Contig {
            name: "chr2".into(),
            length: 500_000,
        },
    ])
    .unwrap();

    let _store = PbzStore::create(&path, genome, Some("GRCh38".into())).unwrap();

    // Zarr v3 store on disk
    assert!(path.join("zarr.json").exists());
    assert!(path.join("contigs").join("zarr.json").exists());
    assert!(path.join("contig_lengths").join("zarr.json").exists());
}

#[test]
fn open_reads_back_genome_and_coordinate_space() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.pbz");

    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 1_000_000,
        },
        Contig {
            name: "chrX".into(),
            length: 155_270_560,
        },
    ])
    .unwrap();

    {
        let _ = PbzStore::create(&path, genome, Some("GRCh38".into())).unwrap();
    }

    let store = PbzStore::open(&path).unwrap();
    assert_eq!(store.genome().len(), 2);
    assert_eq!(store.genome().contigs()[0].name, "chr1");
    assert_eq!(store.genome().contigs()[1].length, 155_270_560);
    assert_eq!(store.coordinate_space(), Some("GRCh38"));
    assert_eq!(store.track_names().count(), 0);
}
