use ndarray::ArrayViewMut2;
use pbzarr::import::{Config, run_pipeline};
use pbzarr::io::Dtype;
use pbzarr::io::ValueReader;
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};
use tempfile::TempDir;

/// A reader that fills every position with a constant value.
struct ConstReader {
    genome: Genome,
    val: u32,
}

impl ValueReader for ConstReader {
    type Item = u32;

    fn contigs(&self) -> &Genome {
        &self.genome
    }

    fn n_fields(&self) -> usize {
        1
    }

    fn read_into(
        &self,
        _contig_name: &str,
        _start: u64,
        _end: u64,
        mut dst: ArrayViewMut2<'_, u32>,
    ) -> pbzarr::io::Result<()> {
        dst.fill(self.val);
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(Self {
            genome: self.genome.clone(),
            val: self.val,
        })
    }
}

#[test]
fn pipeline_writes_constants_into_cohort_track() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 5_000,
    }])
    .unwrap();

    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome.clone(),
            TrackConfig::new(Dtype::U32)
                .columns(vec!["A".into(), "B".into()])
                .column_dim("sample"),
        )
        .unwrap();

    let readers = vec![
        ConstReader {
            genome: genome.clone(),
            val: 7,
        },
        ConstReader {
            genome: genome.clone(),
            val: 13,
        },
    ];

    let track = store.track("depth").unwrap();
    let report = run_pipeline::<u32, _>(track, readers, &Config::default()).unwrap();
    assert!(report.bytes_written > 0);
    assert_eq!(report.contigs_written, 1);

    let region = Region {
        contig: genome.id("chr1").unwrap(),
        start: 0,
        end: 5_000,
    };
    let got = store
        .track("depth")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    let got2 = got.into_dimensionality::<ndarray::Ix2>().unwrap();
    for i in 0..5_000 {
        assert_eq!(got2[[i, 0]], 7, "col 0 at i={i}");
        assert_eq!(got2[[i, 1]], 13, "col 1 at i={i}");
    }
}

#[test]
fn pipeline_writes_constants_into_scalar_track() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 3_000,
    }])
    .unwrap();

    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("mask", genome.clone(), TrackConfig::new(Dtype::U32))
        .unwrap();

    let readers = vec![ConstReader {
        genome: genome.clone(),
        val: 42,
    }];

    let track = store.track("mask").unwrap();
    let report = run_pipeline::<u32, _>(track, readers, &Config::default()).unwrap();
    assert!(report.bytes_written > 0);

    let region = Region {
        contig: genome.id("chr1").unwrap(),
        start: 0,
        end: 3_000,
    };
    let got = store
        .track("mask")
        .unwrap()
        .read_region::<u32>(&region)
        .unwrap();
    let got1 = got.into_dimensionality::<ndarray::Ix1>().unwrap();
    assert!(got1.iter().all(|&v| v == 42));
}

#[test]
fn pipeline_spans_multiple_contigs() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("m.pbz");
    let genome = Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 2_000,
        },
        Contig {
            name: "chr2".into(),
            length: 1_500,
        },
    ])
    .unwrap();

    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "depth",
            genome.clone(),
            TrackConfig::new(Dtype::U32)
                .columns(vec!["S".into()])
                .column_dim("sample"),
        )
        .unwrap();

    let readers = vec![ConstReader {
        genome: genome.clone(),
        val: 99,
    }];
    let track = store.track("depth").unwrap();
    let report = run_pipeline::<u32, _>(track, readers, &Config::default()).unwrap();
    assert_eq!(report.contigs_written, 2);

    for (name, len) in [("chr1", 2_000usize), ("chr2", 1_500)] {
        let region = Region {
            contig: genome.id(name).unwrap(),
            start: 0,
            end: len as u64,
        };
        let got = store
            .track("depth")
            .unwrap()
            .read_region::<u32>(&region)
            .unwrap();
        let got2 = got.into_dimensionality::<ndarray::Ix2>().unwrap();
        for i in 0..len {
            assert_eq!(got2[[i, 0]], 99, "{name} i={i}");
        }
    }
}
