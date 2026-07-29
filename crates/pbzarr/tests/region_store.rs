//! Region-store builder round-trips: scalar + 2D sources, regions that straddle
//! output chunk boundaries, gathered values match the source slices.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2};
use pbzarr::genome::{Contig, Genome};
use pbzarr::io::Dtype;
use pbzarr::region_store::{RegionBuildConfig, build_region_store};
use pbzarr::{PbzStore, Region, TrackConfig};
use tempfile::TempDir;

fn genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 50,
        },
        Contig {
            name: "chr2".into(),
            length: 30,
        },
    ])
    .unwrap()
}

/// [("chr1",10,20), ("chr2",5,15), ("chr1",0,5)] — deliberately unsorted, no
/// overlaps. Sorted flat order is (chr1,0-5), (chr1,10-20), (chr2,5-15).
fn intervals() -> Vec<(String, u64, u64)> {
    vec![
        ("chr1".into(), 10, 20),
        ("chr2".into(), 5, 15),
        ("chr1".into(), 0, 5),
    ]
}

#[test]
fn scalar_region_build_gathers_source_slices() {
    let dir = TempDir::new().unwrap();
    let g = genome();

    // Source scalar track: value == flat position.
    let mut src = PbzStore::create(dir.path().join("src.pbz")).unwrap();
    src.create_track(
        "depth",
        g.clone(),
        TrackConfig::new(Dtype::I32).chunk_size(16),
    )
    .unwrap();
    {
        let t = src.track("depth").unwrap();
        let c1 = Region {
            contig: g.id("chr1").unwrap(),
            start: 0,
            end: 50,
        };
        t.write_region::<i32>(&c1, Array1::from_iter(0..50i32).into_dyn())
            .unwrap();
        let c2 = Region {
            contig: g.id("chr2").unwrap(),
            start: 0,
            end: 30,
        };
        t.write_region::<i32>(&c2, Array1::from_iter(50..80i32).into_dyn())
            .unwrap();
    }

    // Small output chunk so region 1 (flat [5,15)) straddles a chunk boundary.
    let mut out = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    build_region_store(
        Arc::new(src),
        &intervals(),
        &mut out,
        RegionBuildConfig {
            chunk_size: Some(8),
            workers: 3,
            ..Default::default()
        },
    )
    .unwrap();

    // Region contigs are index strings in sorted order.
    let expected: [Vec<i32>; 3] = [
        (0..5).collect(),   // chr1 0-5
        (10..20).collect(), // chr1 10-20
        (55..65).collect(), // chr2 5-15 -> flat 55..65
    ];
    let t = out.track("depth").unwrap();
    let rg = t.genome();
    for (i, want) in expected.iter().enumerate() {
        let r = Region {
            contig: rg.id(&i.to_string()).unwrap(),
            start: 0,
            end: want.len() as u64,
        };
        let got = t
            .read_region::<i32>(&r)
            .unwrap()
            .into_dimensionality::<Ix1>()
            .unwrap();
        assert_eq!(got.to_vec(), *want, "region {i}");
    }
}

#[test]
fn two_d_region_build_fills_all_columns_from_one_reader() {
    let dir = TempDir::new().unwrap();
    let g = genome();

    // Source 2D track (2 columns): value[pos, col] == pos*10 + col.
    let mut src = PbzStore::create(dir.path().join("src2d.pbz")).unwrap();
    src.create_track(
        "cov",
        g.clone(),
        TrackConfig::new(Dtype::I32)
            .columns(vec!["a".into(), "b".into()])
            .column_dim("sample")
            .chunk_size(16),
    )
    .unwrap();
    {
        let t = src.track("cov").unwrap();
        for (name, base, len) in [("chr1", 0u64, 50usize), ("chr2", 50, 30)] {
            let mut a = Array2::<i32>::zeros((len, 2));
            for p in 0..len {
                a[[p, 0]] = ((base as usize + p) * 10) as i32;
                a[[p, 1]] = ((base as usize + p) * 10 + 1) as i32;
            }
            let r = Region {
                contig: g.id(name).unwrap(),
                start: 0,
                end: len as u64,
            };
            t.write_region::<i32>(&r, a.into_dyn()).unwrap();
        }
    }

    let mut out = PbzStore::create(dir.path().join("out2d.pbz")).unwrap();
    build_region_store(
        Arc::new(src),
        &intervals(),
        &mut out,
        RegionBuildConfig {
            chunk_size: Some(8),
            workers: 3,
            ..Default::default()
        },
    )
    .unwrap();

    let t = out.track("cov").unwrap();
    assert_eq!(t.rank(), 2);
    assert_eq!(t.column_dim(), Some("sample"));
    let rg = t.genome();

    // Region 2 = (chr2, 5-15) -> source flat positions 55..65.
    let r2: ArrayD<i32> = t
        .read_region::<i32>(&Region {
            contig: rg.id("2").unwrap(),
            start: 0,
            end: 10,
        })
        .unwrap();
    let r2 = r2.into_dimensionality::<Ix2>().unwrap();
    for row in 0..10usize {
        let src_pos = 55 + row;
        assert_eq!(r2[[row, 0]], (src_pos * 10) as i32);
        assert_eq!(r2[[row, 1]], (src_pos * 10 + 1) as i32);
    }
}

#[test]
fn overlapping_intervals_rejected() {
    let dir = TempDir::new().unwrap();
    let g = genome();
    let mut src = PbzStore::create(dir.path().join("src.pbz")).unwrap();
    src.create_track(
        "depth",
        g.clone(),
        TrackConfig::new(Dtype::I32).chunk_size(16),
    )
    .unwrap();
    src.track("depth")
        .unwrap()
        .write_region::<i32>(
            &Region {
                contig: g.id("chr1").unwrap(),
                start: 0,
                end: 50,
            },
            Array1::from_iter(0..50i32).into_dyn(),
        )
        .unwrap();

    let mut out = PbzStore::create(dir.path().join("out.pbz")).unwrap();
    let overlap = vec![("chr1".into(), 0, 20), ("chr1".into(), 10, 30)];
    let err = build_region_store(
        Arc::new(src),
        &overlap,
        &mut out,
        RegionBuildConfig::default(),
    );
    assert!(err.is_err(), "overlapping intervals must be rejected");
}

#[test]
fn source_chunk_builder_handles_parallel_writer_boundaries() {
    let dir = TempDir::new().unwrap();
    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 64,
    }])
    .unwrap();

    let mut src = PbzStore::create(dir.path().join("src_parallel.pbz")).unwrap();
    src.create_track(
        "cov",
        g.clone(),
        TrackConfig::new(Dtype::I32)
            .columns(vec!["a".into(), "b".into(), "c".into()])
            .column_dim("column")
            .chunk_size(10),
    )
    .unwrap();
    {
        let t = src.track("cov").unwrap();
        let data = Array2::from_shape_fn((64, 3), |(r, c)| (r as i32 * 100) + c as i32);
        t.write_region::<i32>(
            &Region {
                contig: g.id("chr1").unwrap(),
                start: 0,
                end: 64,
            },
            data.into_dyn(),
        )
        .unwrap();
    }

    let intervals = vec![
        ("chr1".to_owned(), 8, 13),
        ("chr1".to_owned(), 19, 27),
        ("chr1".to_owned(), 33, 45),
    ];
    let mut out = PbzStore::create(dir.path().join("out_parallel.pbz")).unwrap();

    build_region_store(
        Arc::new(src),
        &intervals,
        &mut out,
        RegionBuildConfig {
            chunk_size: Some(7),
            workers: 6,
            write_workers: Some(3),
            decode_workers: Some(3),
            ..Default::default()
        },
    )
    .unwrap();

    let t = out.track("cov").unwrap();
    let rg = t.genome();
    let r0 = t
        .read_region::<i32>(&Region {
            contig: rg.id("0").unwrap(),
            start: 0,
            end: 5,
        })
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert_eq!(r0[[0, 0]], 800);
    assert_eq!(r0[[4, 2]], 1202);

    let r1 = t
        .read_region::<i32>(&Region {
            contig: rg.id("1").unwrap(),
            start: 0,
            end: 8,
        })
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert_eq!(r1[[0, 0]], 1900);
    assert_eq!(r1[[7, 2]], 2602);

    let r2 = t
        .read_region::<i32>(&Region {
            contig: rg.id("2").unwrap(),
            start: 0,
            end: 12,
        })
        .unwrap()
        .into_dimensionality::<Ix2>()
        .unwrap();
    assert_eq!(r2[[0, 0]], 3300);
    assert_eq!(r2[[11, 2]], 4402);
}

#[test]
fn region_build_works_with_memory_store_outputs() {
    use zarrs::storage::ReadableWritableListableStorage;
    use zarrs::storage::store::MemoryStore;

    let g = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 20,
    }])
    .unwrap();

    let src_storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let mut src = PbzStore::create_with_storage(src_storage).unwrap();
    src.create_track(
        "depth",
        g.clone(),
        TrackConfig::new(Dtype::I32).chunk_size(6),
    )
    .unwrap();
    src.track("depth")
        .unwrap()
        .write_region::<i32>(
            &Region {
                contig: g.id("chr1").unwrap(),
                start: 0,
                end: 20,
            },
            Array1::from_iter(0..20i32).into_dyn(),
        )
        .unwrap();

    let out_storage: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
    let mut out = PbzStore::create_with_storage(out_storage).unwrap();
    build_region_store(
        Arc::new(src),
        &[("chr1".to_owned(), 4, 13)],
        &mut out,
        RegionBuildConfig {
            chunk_size: Some(5),
            workers: 4,
            write_workers: Some(2),
            decode_workers: Some(2),
            ..Default::default()
        },
    )
    .unwrap();

    let t = out.track("depth").unwrap();
    let rg = t.genome();
    let got = t
        .read_region::<i32>(&Region {
            contig: rg.id("0").unwrap(),
            start: 0,
            end: 9,
        })
        .unwrap()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert_eq!(got.to_vec(), (4..13).collect::<Vec<i32>>());
}
