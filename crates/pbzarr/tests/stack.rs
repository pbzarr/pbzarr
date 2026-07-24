//! Batch stack: N single-sample stores -> one cohort store.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array1, ArrayD, Ix2};
use pbzarr::genome::Contig;
use pbzarr::import::ProgressSink;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore, StackConfig, TrackConfig, stack};
use tempfile::TempDir;

fn genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 100,
        },
        Contig {
            name: "chr2".into(),
            length: 50,
        },
    ])
    .unwrap()
}

/// A single-sample store with a scalar `depth` (i32) and `score` (f32) track,
/// both filled on chr1[0,100) with `value` / `value*0.5`.
fn make_sample(path: &std::path::Path, value: i32) {
    let mut store = PbzStore::create(path).unwrap();
    store
        .create_track("depth", genome(), TrackConfig::new(Dtype::I32))
        .unwrap();
    store
        .create_track("score", genome(), TrackConfig::new(Dtype::F32))
        .unwrap();

    let region = store
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:0-100".parse().unwrap())
        .unwrap();
    store
        .track("depth")
        .unwrap()
        .write_region(&region, Array1::from(vec![value; 100]).into_dyn())
        .unwrap();
    store
        .track("score")
        .unwrap()
        .write_region(
            &region,
            Array1::from(vec![value as f32 * 0.5; 100]).into_dyn(),
        )
        .unwrap();
}

#[test]
fn stack_two_samples_into_cohort() {
    let dir = TempDir::new().unwrap();
    let p1 = dir.path().join("s1.pbz");
    let p2 = dir.path().join("s2.pbz");
    make_sample(&p1, 3);
    make_sample(&p2, 7);

    let s1 = PbzStore::open(&p1).unwrap();
    let s2 = PbzStore::open(&p2).unwrap();
    let mut out = PbzStore::create(dir.path().join("cohort.pbz")).unwrap();
    stack(
        vec![(s1, "s1".into()), (s2, "s2".into())],
        &mut out,
        StackConfig::default(),
    )
    .unwrap();

    let depth = out.track("depth").unwrap();
    assert_eq!(depth.rank(), 2);
    assert_eq!(depth.column_dim(), Some("sample"));
    assert_eq!(depth.columns_count().unwrap(), 2);

    let region = out
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr1:0-100".parse().unwrap())
        .unwrap();
    let arr: ArrayD<i32> = depth.read_region(&region).unwrap();
    let a = arr.into_dimensionality::<Ix2>().unwrap();
    assert_eq!(a.shape(), &[100, 2]);
    assert!(a.column(0).iter().all(|&v| v == 3));
    assert!(a.column(1).iter().all(|&v| v == 7));

    // chr2 was never written in the sources -> cohort fill (0).
    let chr2 = out
        .genome_for("depth")
        .unwrap()
        .resolve(&"chr2:0-50".parse().unwrap())
        .unwrap();
    let chr2 = depth.read_region::<i32>(&chr2).unwrap();
    assert!(chr2.iter().all(|&v| v == 0));

    // float track stacks too, with its own dtype.
    let score = out.track("score").unwrap();
    assert_eq!(score.dtype(), Dtype::F32);
    let sr = out
        .genome_for("score")
        .unwrap()
        .resolve(&"chr1:0-100".parse().unwrap())
        .unwrap();
    let sa: ArrayD<f32> = score.read_region(&sr).unwrap();
    let sa = sa.into_dimensionality::<Ix2>().unwrap();
    assert!(sa.column(0).iter().all(|&v| v == 1.5));
    assert!(sa.column(1).iter().all(|&v| v == 3.5));
}

#[derive(Default)]
struct CountingSink {
    bytes: AtomicU64,
    dones: AtomicU64,
}

impl ProgressSink for CountingSink {
    fn tick(&self, bytes: u64) {
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }
    fn done(&self) {
        self.dones.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn stack_reports_progress_per_track() {
    let dir = TempDir::new().unwrap();
    let p1 = dir.path().join("s1.pbz");
    let p2 = dir.path().join("s2.pbz");
    make_sample(&p1, 3);
    make_sample(&p2, 7);

    let s1 = PbzStore::open(&p1).unwrap();
    let s2 = PbzStore::open(&p2).unwrap();
    let mut out = PbzStore::create(dir.path().join("cohort.pbz")).unwrap();

    let sink = Arc::new(CountingSink::default());
    let seen = Arc::clone(&sink);
    let cfg = StackConfig {
        progress: Some(Arc::new(move |_label: &str, _total: u64| {
            Arc::clone(&seen) as Arc<dyn ProgressSink>
        })),
        ..StackConfig::default()
    };
    stack(vec![(s1, "s1".into()), (s2, "s2".into())], &mut out, cfg).unwrap();

    // Two scalar tracks (depth i32, score f32), genome ΣL=150, 2 samples.
    // depth: 150*2*4, score: 150*2*4 -> 2400 bytes total; done() once per track.
    assert_eq!(sink.bytes.load(Ordering::Relaxed), 150 * 2 * 4 * 2);
    assert_eq!(sink.dones.load(Ordering::Relaxed), 2);
}

#[test]
fn stack_rejects_genome_mismatch() {
    let dir = TempDir::new().unwrap();
    let p1 = dir.path().join("s1.pbz");
    make_sample(&p1, 3);

    // A second store with a different genome (chr1 only).
    let p2 = dir.path().join("s2.pbz");
    let mut s2 = PbzStore::create(&p2).unwrap();
    let g2 = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 100,
    }])
    .unwrap();
    s2.create_track("depth", g2, TrackConfig::new(Dtype::I32))
        .unwrap();
    drop(s2);

    let a = PbzStore::open(&p1).unwrap();
    let b = PbzStore::open(&p2).unwrap();
    let mut out = PbzStore::create(dir.path().join("cohort.pbz")).unwrap();
    let err = stack(
        vec![(a, "s1".into()), (b, "s2".into())],
        &mut out,
        StackConfig {
            tracks: Some(vec!["depth".into()]),
            ..StackConfig::default()
        },
    );
    assert!(err.is_err(), "expected genome-mismatch error");
}
