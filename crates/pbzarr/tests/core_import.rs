use pbzarr::Contig;
use pbzarr::dataset::{Dataset, Reference, Role, ValidationStatus};
use std::sync::Arc;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

fn reference() -> Reference {
    Reference::new(vec![
        Contig {
            name: "chr1".into(),
            length: 5,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap()
}
#[test]
fn creates_shared_reference() {
    let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let ds = Dataset::create_with_storage(storage.clone(), reference()).unwrap();
    assert_eq!(ds.reference().unwrap().contigs(), reference().contigs());
    assert_eq!(ds.variables()["contig_length"].role, Role::Coordinate);
    assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    let reopened = Dataset::open_with_storage(storage).unwrap();
    assert_eq!(
        reopened
            .read_raw::<i64>("contig_length", &[0..2])
            .unwrap()
            .as_slice()
            .unwrap(),
        &[5, 7]
    );
}
#[test]
fn creates_empty_and_zero_length_references() {
    for contigs in [
        vec![],
        vec![Contig {
            name: "empty".into(),
            length: 0,
        }],
    ] {
        let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let ds = Dataset::create_with_storage(storage, Reference::new(contigs.clone()).unwrap())
            .unwrap();
        assert_eq!(ds.reference().unwrap().contigs(), contigs);
        assert_eq!(ds.validate().unwrap().status, ValidationStatus::Valid);
    }
}
#[test]
fn creation_rejects_occupied_namespaces_without_writing() {
    for key in ["orphan/c/0", "bare", "zarr.json"] {
        let storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
        let key = StoreKey::new(key).unwrap();
        storage.set(&key, b"untouched".to_vec().into()).unwrap();
        assert!(Dataset::create_with_storage(storage.clone(), reference()).is_err());
        assert_eq!(storage.get(&key).unwrap().unwrap().as_ref(), b"untouched");
        assert!(
            storage
                .get(&StoreKey::new("contig/zarr.json").unwrap())
                .unwrap()
                .is_none()
        );
    }
}
