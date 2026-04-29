//! Test fixtures shared across e2e CLI tests.
//!
//! All fixtures are synthesized in-process — no `.pbz.zarr` directories
//! or `.d4` files are committed to the repo.

#![allow(dead_code)] // each test binary uses a subset

use ndarray::Array2;
use pbzarr::{PbzStore, TrackConfig};
use std::path::PathBuf;
use tempfile::TempDir;

pub struct StoreFixture {
    pub _dir: TempDir,
    pub path: PathBuf,
}

impl StoreFixture {
    /// Create a store with chr1=1000bp, chr2=500bp and no tracks.
    pub fn empty() -> Self {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("s.pbz.zarr");
        PbzStore::create(
            &path,
            &["chr1".to_string(), "chr2".to_string()],
            &[1000u64, 500u64],
        )
        .unwrap();
        Self { _dir: dir, path }
    }

    /// Create a store and add a u32 columnar track named `name` with `num_samples`
    /// columns each filled with `value`.
    pub fn with_uint32_track(name: &str, value: u32, num_samples: usize) -> Self {
        let f = Self::empty();
        let store = PbzStore::open(&f.path).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: Some((0..num_samples).map(|i| format!("s{i}")).collect()),
            chunk_size: 100,
            column_chunk_size: 4,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = store.create_track(name, cfg).unwrap();
        let chr1 = Array2::<u32>::from_elem((1000, num_samples), value);
        for chunk_idx in 0..10 {
            let start = chunk_idx * 100;
            let chunk = chr1.slice(ndarray::s![start..start + 100, ..]).to_owned();
            track.write_chunk("chr1", chunk_idx as u64, chunk).unwrap();
        }
        let chr2 = Array2::<u32>::from_elem((500, num_samples), value);
        for chunk_idx in 0..5 {
            let start = chunk_idx * 100;
            let chunk = chr2.slice(ndarray::s![start..start + 100, ..]).to_owned();
            track.write_chunk("chr2", chunk_idx as u64, chunk).unwrap();
        }
        f
    }

    /// Create a store with a single-column u32 track for bedGraph testing.
    pub fn with_single_col_uint32(name: &str) -> Self {
        Self::with_uint32_track(name, 7, 1)
    }

    /// Create a store with a bool scalar track for BED testing.
    pub fn with_bool_track(name: &str) -> Self {
        let f = Self::empty();
        let store = PbzStore::open(&f.path).unwrap();
        let cfg = TrackConfig {
            dtype: "bool".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = store.create_track(name, cfg).unwrap();
        // Write all-true for chr1 first chunk (positions 0..100), default-false elsewhere.
        let true_chunk = ndarray::Array1::<bool>::from_elem(100, true);
        track.write_chunk_1d("chr1", 0, true_chunk).unwrap();
        f
    }
}
