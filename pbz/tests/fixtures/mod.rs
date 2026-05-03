//! Test fixtures shared across e2e CLI tests.
//!
//! All fixtures are synthesized in-process — no `.pbz.zarr` directories
//! or `.d4` files are committed to the repo.

#![allow(dead_code)] // each test binary uses a subset

use ndarray::{Array1, Array2};
use noodles::bgzf;
use pbzarr::{PbzStore, TrackConfig};
use std::io::Write;
use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};

/// Build a bgzipped BED file in a tempfile from the given BED text. Returns the
/// tempfile (caller must keep it alive while the path is in use).
pub fn build_bgz_bed(text: &str) -> NamedTempFile {
    let f = tempfile::Builder::new()
        .prefix("test-bed-")
        .suffix(".bed.gz")
        .tempfile()
        .unwrap();
    let mut writer = bgzf::io::Writer::new(std::fs::File::create(f.path()).unwrap());
    writer.write_all(text.as_bytes()).unwrap();
    writer.finish().unwrap();
    f
}

/// Write a chrom.sizes / FAI file (2-column tab-separated). Returns the tempfile.
pub fn build_chrom_sizes(contigs: &[(&str, u64)]) -> NamedTempFile {
    let f = tempfile::Builder::new()
        .prefix("contigs-")
        .suffix(".txt")
        .tempfile()
        .unwrap();
    let mut s = String::new();
    for (name, len) in contigs {
        s.push_str(&format!("{name}\t{len}\n"));
    }
    std::fs::write(f.path(), s).unwrap();
    f
}

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

    /// Create a store and add a u32 columnar track named `name` with
    /// `num_columns` columns each filled with `value`. Requires `num_columns >= 2`
    /// (single-column data must be 1D per spec v0.1).
    pub fn with_uint32_track(name: &str, value: u32, num_columns: usize) -> Self {
        assert!(
            num_columns != 1,
            "with_uint32_track: num_columns=1 is invalid; use with_single_sample_uint32 \
             for the 1D single-column case"
        );
        let f = Self::empty();
        let store = PbzStore::open(&f.path).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: Some((0..num_columns).map(|i| format!("s{i}")).collect()),
            chunk_size: 100,
            column_chunk_size: 4,
            column_dim_name: None,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = store.create_track(name, cfg).unwrap();
        let chr1 = Array2::<u32>::from_elem((1000, num_columns), value);
        for chunk_idx in 0..10usize {
            let start = chunk_idx * 100;
            let chunk = chr1.slice(ndarray::s![start..start + 100, ..]).to_owned();
            track.write_chunk("chr1", chunk_idx as u64, chunk).unwrap();
        }
        let chr2 = Array2::<u32>::from_elem((500, num_columns), value);
        for chunk_idx in 0..5usize {
            let start = chunk_idx * 100;
            let chunk = chr2.slice(ndarray::s![start..start + 100, ..]).to_owned();
            track.write_chunk("chr2", chunk_idx as u64, chunk).unwrap();
        }
        f
    }

    /// Create a store with a single-column (1D) u32 track for bedGraph testing.
    /// Per spec v0.1, single-column data MUST be 1D (no `columns` array).
    pub fn with_single_sample_uint32(name: &str) -> Self {
        let f = Self::empty();
        let store = PbzStore::open(&f.path).unwrap();
        let cfg = TrackConfig {
            dtype: "uint32".into(),
            columns: None,
            chunk_size: 100,
            column_chunk_size: 1,
            column_dim_name: None,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = store.create_track(name, cfg).unwrap();
        let value: u32 = 7;
        for chunk_idx in 0..10u64 {
            let chunk = Array1::<u32>::from_elem(100, value);
            track.write_chunk_1d("chr1", chunk_idx, chunk).unwrap();
        }
        for chunk_idx in 0..5u64 {
            let chunk = Array1::<u32>::from_elem(100, value);
            track.write_chunk_1d("chr2", chunk_idx, chunk).unwrap();
        }
        f
    }

    /// Backwards-compatible alias for tests that haven't been migrated yet.
    pub fn with_single_col_uint32(name: &str) -> Self {
        Self::with_single_sample_uint32(name)
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
            column_dim_name: None,
            description: None,
            source: None,
            extra: serde_json::Map::new(),
        };
        let track = store.create_track(name, cfg).unwrap();
        // Write all-true for chr1 first chunk (positions 0..100), default-false elsewhere.
        let true_chunk = Array1::<bool>::from_elem(100, true);
        track.write_chunk_1d("chr1", 0, true_chunk).unwrap();
        f
    }
}
