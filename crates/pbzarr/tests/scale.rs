//! Multiscale pyramid: level math, metadata, and lifecycle.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD};
use pbzarr::genome::Contig;
use pbzarr::io::Dtype;
use pbzarr::{Genome, PbzStore, ScaleConfig, TrackConfig, scale};
use serde_json::json;
use tempfile::TempDir;
use zarrs::array::{Array, ArraySubset};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorage;

/// chr1 (10) and chr2 (7): neither divisible by factor 4.
fn ragged_genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 10,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap()
}

fn open_array(path: &std::path::Path, node: &str) -> Array<FilesystemStore> {
    let fs = Arc::new(FilesystemStore::new(path).unwrap());
    Array::open(fs, node).unwrap()
}

fn read_all_f32(arr: &Array<FilesystemStore>) -> ArrayD<f32> {
    let ranges: Vec<std::ops::Range<u64>> = arr.shape().iter().map(|&n| 0..n).collect();
    arr.retrieve_array_subset::<ArrayD<f32>>(&ArraySubset::new_with_ranges(&ranges))
        .unwrap()
}

fn factors_config(factors: Vec<u64>) -> ScaleConfig {
    ScaleConfig {
        factors: Some(factors),
        ..ScaleConfig::default()
    }
}

#[test]
fn i32_track_means_with_ragged_last_bins() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", ragged_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let track = store.track("depth").unwrap();
    let r1 = track
        .genome()
        .resolve(&"chr1:0-10".parse().unwrap())
        .unwrap();
    track
        .write_region(&r1, Array1::from((0..10).collect::<Vec<i32>>()).into_dyn())
        .unwrap();
    let r2 = track
        .genome()
        .resolve(&"chr2:0-7".parse().unwrap())
        .unwrap();
    track
        .write_region(
            &r2,
            Array1::from(vec![10i32, 20, 30, 40, 50, 60, 70]).into_dyn(),
        )
        .unwrap();

    let report = scale(&store, "depth", &factors_config(vec![4])).unwrap();
    assert_eq!(report.levels.len(), 1);
    assert_eq!(report.levels[0].factor, 4);
    // Level offsets: ceil(10/4)=3 for chr1, ceil(7/4)=2 for chr2 -> [0, 3, 5].
    assert_eq!(report.levels[0].bins, 5);
    assert!(report.levels[0].bytes_written > 0);

    let level = open_array(&path, "/depth/scales/4/mean");
    assert_eq!(level.shape(), &[5]);
    assert_eq!(
        level.dimension_names().as_ref().unwrap()[0].as_deref(),
        Some("bin")
    );
    let got = read_all_f32(&level).into_raw_vec_and_offset().0;
    // chr1: (0+1+2+3)/4, (4+5+6+7)/4, ragged (8+9)/2; chr2 starts at level
    // offset 3: (10+20+30+40)/4, ragged (50+60+70)/3.
    assert_eq!(got, vec![1.5, 5.5, 8.5, 25.0, 60.0]);
}

#[test]
fn f32_cohort_track_is_nan_aware_per_column() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 8,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "af",
            genome,
            TrackConfig::new(Dtype::F32)
                .columns(vec!["a".into(), "b".into()])
                .column_dim("sample"),
        )
        .unwrap();

    let track = store.track("af").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-8".parse().unwrap())
        .unwrap();
    let nan = f32::NAN;
    let data = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, // col a: 1, 2, NaN, 3 -> mean 2.0
            2.0, 2.0, // col b: 2, 2, 2, 2 -> mean 2.0
            nan, 2.0, //
            3.0, 2.0, //
            nan, 4.0, // col a: all NaN -> NaN
            nan, nan, // col b: 4, NaN, NaN, NaN -> mean 4.0
            nan, nan, //
            nan, nan, //
        ],
    )
    .unwrap();
    track.write_region(&region, data.into_dyn()).unwrap();

    scale(&store, "af", &factors_config(vec![4])).unwrap();

    let level = open_array(&path, "/af/scales/4/mean");
    assert_eq!(level.shape(), &[2, 2]);
    let names = level.dimension_names().as_ref().unwrap().clone();
    assert_eq!(names[0].as_deref(), Some("bin"));
    assert_eq!(names[1].as_deref(), Some("sample"));
    // Level fill is NaN (f32 of the NaN source fill).
    let fill = f32::from_ne_bytes(level.fill_value().as_ne_bytes().try_into().unwrap());
    assert!(fill.is_nan());

    let got = read_all_f32(&level);
    assert_eq!(got[[0, 0]], 2.0);
    assert_eq!(got[[0, 1]], 2.0);
    assert!(got[[1, 0]].is_nan(), "all-NaN bin yields NaN");
    assert_eq!(got[[1, 1]], 4.0);
}

#[test]
fn wide_cohort_bins_are_read_in_column_blocks() {
    // At factor 2^25 one full-width bin (2^25 * 2 cols * 4 B = 256 MiB)
    // exceeds the 64 MiB slab budget, so the engine slabs the column axis
    // (one column per read here). The contig is tiny, so the clamped reads
    // stay small; the per-column means prove each block landed in the right
    // column range of the level.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 6,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track(
            "af",
            genome,
            TrackConfig::new(Dtype::F32).columns(vec!["a".into(), "b".into()]),
        )
        .unwrap();

    let track = store.track("af").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-6".parse().unwrap())
        .unwrap();
    let nan = f32::NAN;
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            4.0, 1.0, // col a: 4, 4 (rest NaN) -> mean 4.0
            4.0, 2.0, // col b: 1..=6 -> mean 3.5
            nan, 3.0, //
            nan, 4.0, //
            nan, 5.0, //
            nan, 6.0, //
        ],
    )
    .unwrap();
    track.write_region(&region, data.into_dyn()).unwrap();

    let factor = 1u64 << 25;
    let report = scale(&store, "af", &factors_config(vec![factor])).unwrap();
    assert_eq!(report.levels[0].bins, 1);

    let level = open_array(&path, &format!("/af/scales/{factor}/mean"));
    assert_eq!(level.shape(), &[1, 2]);
    let got = read_all_f32(&level);
    assert_eq!(got[[0, 0]], 4.0);
    assert_eq!(got[[0, 1]], 3.5);
}

#[test]
fn bool_track_means_are_fraction_true() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 6,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("mask", genome, TrackConfig::new(Dtype::Bool))
        .unwrap();

    let track = store.track("mask").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-6".parse().unwrap())
        .unwrap();
    track
        .write_region(
            &region,
            Array1::from(vec![true, false, true, true, true, false]).into_dyn(),
        )
        .unwrap();

    scale(&store, "mask", &factors_config(vec![4])).unwrap();

    let level = open_array(&path, "/mask/scales/4/mean");
    assert_eq!(level.data_type(), &zarrs::array::data_type::float32());
    let got = read_all_f32(&level).into_raw_vec_and_offset().0;
    assert_eq!(got, vec![0.75, 0.5]);
    // Fill of an all-false bin: 0.0.
    let fill = f32::from_ne_bytes(level.fill_value().as_ne_bytes().try_into().unwrap());
    assert_eq!(fill, 0.0);
}

#[test]
fn publication_writes_binding_metadata_and_preserves_base_attrs() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", ragged_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let storage: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
    let before = zarrs::group::Group::open(storage.clone(), "/depth")
        .unwrap()
        .attributes()
        .clone();

    scale(&store, "depth", &factors_config(vec![4, 8])).unwrap();

    let after = zarrs::group::Group::open(storage, "/depth")
        .unwrap()
        .attributes()
        .clone();

    let transform = |f: u64| {
        json!({"perbase:ragged_axis_scale": {
            "dimension": "position", "factor": f,
            "anchor": "segment-start", "last_bin": "clip"}})
    };
    assert_eq!(
        after["multiscales"],
        json!({"layout": [
            {"asset": "values"},
            {"asset": "scales/4/mean", "derived_from": "values",
             "transform": transform(4), "resampling_method": "average"},
            {"asset": "scales/8/mean", "derived_from": "values",
             "transform": transform(8), "resampling_method": "average"},
        ]})
    );

    // The conventions entry is appended to the existing list.
    let before_conv = before["zarr_conventions"].as_array().unwrap();
    let after_conv = after["zarr_conventions"].as_array().unwrap();
    assert_eq!(after_conv.len(), before_conv.len() + 1);
    assert_eq!(&after_conv[..before_conv.len()], &before_conv[..]);
    assert_eq!(
        *after_conv.last().unwrap(),
        json!({"uuid": "d35379db-88df-4056-af3a-620245f8e347", "name": "multiscales",
               "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1/schema.json",
               "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md"})
    );

    // Base attrs otherwise unchanged.
    let mut stripped = after.clone();
    stripped.remove("multiscales");
    stripped["zarr_conventions"].as_array_mut().unwrap().pop();
    assert_eq!(stripped, before);
}

#[test]
fn lifecycle_rescale_and_sealed_writes_error_and_orphans_are_replaced() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", ragged_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    // Pre-existing scales/ junk without the attr (crashed prior run) is
    // replaced and the run succeeds.
    let junk = path.join("depth/scales/999");
    std::fs::create_dir_all(&junk).unwrap();
    std::fs::write(junk.join("garbage"), b"junk").unwrap();

    scale(&store, "depth", &factors_config(vec![4])).unwrap();
    assert!(!junk.exists(), "orphaned scales/ content is deleted");
    assert!(path.join("depth/scales/4/mean/zarr.json").exists());

    // Second scale call errors (unpublish/rescale is not implemented).
    let err = scale(&store, "depth", &factors_config(vec![4])).unwrap_err();
    assert!(err.to_string().contains("unpublish/rescale"), "{err}");

    // write_region after publication errors on the live handle...
    let track = store.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-4".parse().unwrap())
        .unwrap();
    let err = track
        .write_region(&region, Array1::from(vec![1i32; 4]).into_dyn())
        .unwrap_err();
    assert!(err.to_string().contains("sealed"), "{err}");

    // ...and on a freshly reopened store (seal flag from disk attrs).
    let reopened = PbzStore::open(&path).unwrap();
    let track = reopened.track("depth").unwrap();
    let err = track
        .write_region(&region, Array1::from(vec![1i32; 4]).into_dyn())
        .unwrap_err();
    assert!(err.to_string().contains("sealed"), "{err}");
}

#[test]
fn default_ladder_stops_by_the_2000_bin_rule() {
    let dir = TempDir::new().unwrap();

    // Largest contig 100_000: ceil(100000/32)=3125 > 2000 so the ladder
    // extends to 256; ceil(100000/256)=391 <= 2000 stops it -> [32, 256].
    let path = dir.path().join("big.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 100_000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", genome, TrackConfig::new(Dtype::I32))
        .unwrap();
    let report = scale(&store, "depth", &ScaleConfig::default()).unwrap();
    let factors: Vec<u64> = report.levels.iter().map(|l| l.factor).collect();
    assert_eq!(factors, vec![32, 256]);
    assert_eq!(report.levels[0].bins, 3125);
    assert_eq!(report.levels[1].bins, 391);

    // A tiny genome still gets at least one factor.
    let path = dir.path().join("tiny.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 1000,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", genome, TrackConfig::new(Dtype::I32))
        .unwrap();
    let report = scale(&store, "depth", &ScaleConfig::default()).unwrap();
    let factors: Vec<u64> = report.levels.iter().map(|l| l.factor).collect();
    assert_eq!(factors, vec![32]);
}

/// Naive per-position reference: per contig, per bin of `factor`, the mean
/// of non-NaN values (NaN when a bin has none).
fn naive_means(contigs: &[Vec<f64>], factor: u64) -> Vec<f32> {
    let mut out = Vec::new();
    for vals in contigs {
        for chunk in vals.chunks(factor as usize) {
            let mut s = 0f64;
            let mut c = 0i64;
            for &v in chunk {
                if v.is_nan() {
                    continue;
                }
                s += v;
                c += 1;
            }
            out.push(if c == 0 {
                f32::NAN
            } else {
                (s / c as f64) as f32
            });
        }
    }
    out
}

fn assert_f32_bins_eq(got: &[f32], expect: &[f32], ctx: &str) {
    assert_eq!(got.len(), expect.len(), "{ctx}: length");
    for (i, (g, e)) in got.iter().zip(expect).enumerate() {
        assert!(
            (g.is_nan() && e.is_nan()) || g == e,
            "{ctx}: bin {i}: got {g}, expect {e}"
        );
    }
}

/// chr1 (53) and chr2 (13): divisible by none of 4, 6, 24.
fn cascade_genome() -> Genome {
    Genome::new(vec![
        Contig {
            name: "chr1".into(),
            length: 53,
        },
        Contig {
            name: "chr2".into(),
            length: 13,
        },
    ])
    .unwrap()
}

#[test]
fn cascade_factor_set_matches_naive_binning_i32() {
    // {4, 6, 24}: two roots (4, 6) and one child (24, parent 6); every
    // contig ends mid-bin for every factor, and chr2 is shorter than one
    // aligned unit (lcm(4, 6, 24) = 24).
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", cascade_genome(), TrackConfig::new(Dtype::I32))
        .unwrap();

    let track = store.track("depth").unwrap();
    let chr1: Vec<i32> = (0..53).map(|i| (i * 7) % 23 - 5).collect();
    let chr2: Vec<i32> = (0..13).map(|i| 100 - 9 * i).collect();
    for (name, vals) in [("chr1", &chr1), ("chr2", &chr2)] {
        let region = track
            .genome()
            .resolve(&format!("{name}:0-{}", vals.len()).parse().unwrap())
            .unwrap();
        track
            .write_region(&region, Array1::from(vals.clone()).into_dyn())
            .unwrap();
    }

    let report = scale(&store, "depth", &factors_config(vec![4, 6, 24])).unwrap();
    let as_f64: Vec<Vec<f64>> = [&chr1, &chr2]
        .iter()
        .map(|v| v.iter().map(|&x| f64::from(x)).collect())
        .collect();
    for lvl in &report.levels {
        let f = lvl.factor;
        let level = open_array(&path, &format!("/depth/scales/{f}/mean"));
        let got = read_all_f32(&level).into_raw_vec_and_offset().0;
        let expect = naive_means(&as_f64, f);
        assert_eq!(lvl.bins as usize, expect.len(), "factor {f}: bins");
        assert_eq!(got, expect, "factor {f}");
    }
}

#[test]
fn cascade_factor_set_matches_naive_binning_f32_nan() {
    // Same factor graph on a NaN-fill f32 track with exactly representable
    // values; chr2's first six positions are all NaN, so factor 4's and 6's
    // first chr2 bins are all-NaN and must yield NaN.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("af", cascade_genome(), TrackConfig::new(Dtype::F32))
        .unwrap();

    let track = store.track("af").unwrap();
    let chr1: Vec<f32> = (0..53)
        .map(|i| if i % 5 == 0 { f32::NAN } else { i as f32 })
        .collect();
    let chr2: Vec<f32> = (0..13)
        .map(|i| if i < 6 { f32::NAN } else { (i * 2) as f32 })
        .collect();
    for (name, vals) in [("chr1", &chr1), ("chr2", &chr2)] {
        let region = track
            .genome()
            .resolve(&format!("{name}:0-{}", vals.len()).parse().unwrap())
            .unwrap();
        track
            .write_region(&region, Array1::from(vals.clone()).into_dyn())
            .unwrap();
    }

    let report = scale(&store, "af", &factors_config(vec![4, 6, 24])).unwrap();
    let as_f64: Vec<Vec<f64>> = [&chr1, &chr2]
        .iter()
        .map(|v| v.iter().map(|&x| f64::from(x)).collect())
        .collect();
    for lvl in &report.levels {
        let f = lvl.factor;
        let level = open_array(&path, &format!("/af/scales/{f}/mean"));
        let got = read_all_f32(&level).into_raw_vec_and_offset().0;
        let expect = naive_means(&as_f64, f);
        assert_f32_bins_eq(&got, &expect, &format!("factor {f}"));
    }
}

#[test]
fn huge_lcm_factor_set_falls_back_to_multi_pass() {
    // Two large coprime factors: lcm(2^21, 2^21 + 1) ~ 2^42 blows the Tier A
    // alignment cap, so the per-factor multi-pass fallback runs and must
    // still produce correct output (one ragged bin per factor here).
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz");
    let genome = Genome::new(vec![Contig {
        name: "chr1".into(),
        length: 10,
    }])
    .unwrap();
    let mut store = PbzStore::create(&path).unwrap();
    store
        .create_track("depth", genome, TrackConfig::new(Dtype::I32))
        .unwrap();

    let track = store.track("depth").unwrap();
    let region = track
        .genome()
        .resolve(&"chr1:0-10".parse().unwrap())
        .unwrap();
    track
        .write_region(
            &region,
            Array1::from((0..10).collect::<Vec<i32>>()).into_dyn(),
        )
        .unwrap();

    let f1 = 1u64 << 21;
    let f2 = f1 + 1;
    let report = scale(&store, "depth", &factors_config(vec![f1, f2])).unwrap();
    for (lvl, f) in report.levels.iter().zip([f1, f2]) {
        assert_eq!(lvl.factor, f);
        assert_eq!(lvl.bins, 1);
        let level = open_array(&path, &format!("/depth/scales/{f}/mean"));
        let got = read_all_f32(&level).into_raw_vec_and_offset().0;
        assert_eq!(got, vec![4.5], "factor {f}");
    }
}
