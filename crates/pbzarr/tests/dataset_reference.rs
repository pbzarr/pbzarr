use pbzarr::dataset::{GenomicAxis, Reference};
use pbzarr::{Contig, RegionQuery};
use std::num::NonZeroU64;

fn reference(lengths: &[u64]) -> Reference {
    Reference::new(
        lengths
            .iter()
            .enumerate()
            .map(|(i, &length)| Contig {
                name: format!("chr{}", i + 1),
                length,
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn bins_restart_at_each_contig() {
    let reference = reference(&[5, 7]);
    let axis = GenomicAxis::Bin(NonZeroU64::new(4).unwrap());
    assert_eq!(reference.extent(&axis).unwrap(), 4);
    let query: RegionQuery = "chr2:5-6".parse().unwrap();
    assert_eq!(reference.rows(&axis, &query).unwrap(), 3..4);
    let interval = reference.interval(&axis, 3).unwrap();
    assert_eq!(
        (interval.contig.as_str(), interval.start, interval.stop),
        ("chr2", 4, 7)
    );
    assert_eq!(
        reference.rows(&axis, &"chr2:0-4".parse().unwrap()).unwrap(),
        2..3
    );
    assert_eq!(
        reference.rows(&GenomicAxis::Position, &query).unwrap(),
        10..11
    );
}

#[test]
fn rejects_reference_extent_overflow_and_invalid_names() {
    assert!(
        Reference::new(vec![
            Contig {
                name: "a".into(),
                length: i64::MAX as u64
            },
            Contig {
                name: "b".into(),
                length: 1
            },
        ])
        .is_err()
    );
    assert!(
        Reference::new(vec![Contig {
            name: "a".into(),
            length: u64::MAX
        }])
        .is_err()
    );
    assert!(
        Reference::new(vec![Contig {
            name: "".into(),
            length: 0
        }])
        .is_err()
    );
    assert!(
        Reference::new(vec![
            Contig {
                name: "a".into(),
                length: 1
            },
            Contig {
                name: "a".into(),
                length: 1
            },
        ])
        .is_err()
    );
}

#[test]
fn empty_contigs_and_large_widths_preserve_geometry() {
    let reference = reference(&[0, 5, 0]);
    let axis = GenomicAxis::Bin(NonZeroU64::new(u64::MAX).unwrap());
    assert_eq!(reference.extent(&axis).unwrap(), 1);
    let interval = reference.interval(&axis, 0).unwrap();
    assert_eq!(
        (interval.contig.as_str(), interval.start, interval.stop),
        ("chr2", 0, 5)
    );
    assert!(reference.interval(&axis, 1).is_err());
    assert!(reference.rows(&axis, &"chr1".parse().unwrap()).is_err());
    let empty = Reference::new(vec![]).unwrap();
    assert_eq!(empty.extent(&GenomicAxis::Position).unwrap(), 0);
    assert!(empty.interval(&GenomicAxis::Position, 0).is_err());
    assert_eq!(
        reference
            .extent(&GenomicAxis::Bin(NonZeroU64::new(1).unwrap()))
            .unwrap(),
        5
    );
}

#[test]
fn invalid_queries_do_not_get_rounded_into_a_bin() {
    let reference = reference(&[5]);
    let axis = GenomicAxis::Bin(NonZeroU64::new(4).unwrap());
    for query in [
        RegionQuery {
            contig: "chr1".into(),
            start: Some(2),
            end: Some(2),
        },
        RegionQuery {
            contig: "chr1".into(),
            start: Some(2),
            end: Some(6),
        },
        RegionQuery {
            contig: "chr2".into(),
            start: None,
            end: None,
        },
    ] {
        assert!(reference.rows(&axis, &query).is_err());
    }
}
