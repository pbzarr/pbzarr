//! Single-file zip stores written through the import path.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use pbzarr::import::Import;
use pbzarr::io::{Dtype, OutputSchema, OutputSinkMut, ValueReader};
use pbzarr::{Contig, Dataset, Reference, RegionQuery, TrackConfig};
use rc_zip::fsm::{ArchiveFsm, FsmResult};
use rc_zip::parse::{Archive, Method};
use tempfile::TempDir;
use zarrs::storage::StoreKey;
use zarrs_zip::ZipStorageAdapter;

/// Writes 1 on chr1 and 0 (the fill value) on chr2.
#[derive(Clone)]
struct ContigReader {
    genome: Reference,
    schema: OutputSchema,
}

impl ValueReader for ContigReader {
    fn contigs(&self) -> &Reference {
        &self.genome
    }

    fn output_schema(&self) -> &OutputSchema {
        &self.schema
    }

    fn read_into(
        &mut self,
        contig: &str,
        _start: u64,
        _end: u64,
        outputs: &mut [OutputSinkMut<'_>],
    ) -> pbzarr::io::Result<()> {
        let value = i32::from(contig == "chr1");
        match &mut outputs[0] {
            OutputSinkMut::I32(dst) => dst.fill(value),
            OutputSinkMut::F32(dst) => dst.fill(value as f32),
            _ => panic!("test reader supports i32/f32 sinks only"),
        }
        Ok(())
    }

    fn fork(&self) -> pbzarr::io::Result<Self> {
        Ok(self.clone())
    }
}

fn import(ds: &mut Dataset, name: &str, cfg: TrackConfig) {
    let reader = ContigReader {
        genome: ds.reference().unwrap().clone(),
        schema: OutputSchema::single("value", cfg.dtype),
    };
    let width = cfg.columns.as_ref().map_or(1, Vec::len);
    let builder = Import::from_readers(vec![reader; width])
        .unwrap()
        .into_dataset(ds, vec![(name.into(), cfg.clone())]);
    let builder = if cfg.columns.is_some() {
        builder.readers_as_columns()
    } else {
        builder
    };
    builder.run().unwrap();
}

/// chr1 fills the first shard and part of the second; the last shard is
/// chr2 only, so its values are all fill and the engine erases it.
fn write_zip(path: &Path) {
    let reference = Reference::new(vec![
        Contig {
            name: "chr1".into(),
            length: 10,
        },
        Contig {
            name: "chr2".into(),
            length: 7,
        },
    ])
    .unwrap();
    let mut ds = Dataset::create_zip(path, reference).unwrap();
    import(
        &mut ds,
        "af",
        TrackConfig::new(Dtype::F32)
            .columns(vec!["s1".into(), "s2".into()])
            .column_dim("sample")
            .chunk_size(4)
            .shard_size(8),
    );
    import(
        &mut ds,
        "depth",
        TrackConfig::new(Dtype::I32).chunk_size(4).shard_size(8),
    );
    assert!(!path.exists(), "the archive appears only at close");
    ds.close().unwrap();
}

fn parse(bytes: &[u8]) -> Archive {
    let mut fsm = ArchiveFsm::new(bytes.len() as u64);
    loop {
        if let Some(offset) = fsm.wants_read() {
            let offset = offset as usize;
            let space = fsm.space();
            let n = space.len().min(bytes.len() - offset);
            space[..n].copy_from_slice(&bytes[offset..offset + n]);
            fsm.fill(n);
        }
        match fsm.process().unwrap() {
            FsmResult::Continue(next) => fsm = next,
            FsmResult::Done(archive) => return archive,
        }
    }
}

fn files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect()
}

#[test]
fn an_imported_zip_store_opens_read_only_and_reads_back() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz.zip");
    write_zip(&path);
    assert_eq!(
        files(dir.path()),
        std::slice::from_ref(&path),
        "no temporary file is left"
    );

    let fs = Arc::new(zarrs::filesystem::FilesystemStore::new(dir.path()).unwrap());
    let zip = ZipStorageAdapter::new(fs, StoreKey::new("s.pbz.zip").unwrap()).unwrap();
    let ds = Dataset::open_readable(Arc::new(zip)).unwrap();
    let names: Vec<&str> = ds.variables().keys().map(String::as_str).collect();
    assert_eq!(
        names,
        ["af", "contig_id", "contig_length", "depth", "sample"]
    );
    let chr1: RegionQuery = "chr1".parse().unwrap();
    let af = ds.read_region::<f32>("af", &chr1).unwrap();
    assert_eq!(af.shape(), [10, 2]);
    assert!(af.iter().all(|&v| v == 1.0));
    let chr2: RegionQuery = "chr2".parse().unwrap();
    let depth = ds.read_region::<i32>("depth", &chr2).unwrap();
    assert_eq!(depth.shape(), [7]);
    assert!(depth.iter().all(|&v| v == 0));
}

#[test]
fn a_zip_store_follows_the_archive_profile() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz.zip");
    write_zip(&path);

    let archive = parse(&std::fs::read(&path).unwrap());
    let names: Vec<&str> = archive.entries().map(|e| e.name.as_str()).collect();
    assert_eq!(names[0], "zarr.json");
    assert!(archive.entries().all(|e| e.method == Method::Store));
    let unique: BTreeSet<&str> = names.iter().copied().collect();
    assert_eq!(unique.len(), names.len(), "duplicate names: {names:?}");
    assert!(names.contains(&"af/c/0/0"));
    assert!(
        !names.contains(&"depth/c/2"),
        "an all-fill shard is not written: {names:?}"
    );
    run_python_check(&path);
}

#[test]
fn a_zip_store_dropped_before_close_leaves_no_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.pbz.zip");
    let reference = Reference::new(vec![Contig {
        name: "chr1".into(),
        length: 10,
    }])
    .unwrap();
    let ds = Dataset::create_zip(&path, reference).unwrap();
    drop(ds);
    assert!(files(dir.path()).is_empty());
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .unwrap()
}

const PYTHON_CHECK: &str = r#"
import sys
import xarray as xr
import zarr

store = zarr.storage.ZipStore(sys.argv[1], mode="r")
ds = xr.open_zarr(store, consolidated=True)
assert set(ds.data_vars) >= {"af", "depth"}, ds
assert float(ds["af"].sum()) == 20.0, ds["af"].values
assert int(ds["depth"].sum()) == 10, ds["depth"].values
"#;

/// zarr-python reads the zip through its consolidated metadata. Skips
/// (loudly) when pixi, zarr or xarray is unavailable.
fn run_python_check(path: &Path) {
    let root = workspace_root();
    let available = Command::new("pixi")
        .current_dir(&root)
        .args(["run", "python", "-c", "import xarray, zarr"])
        .output()
        .is_ok_and(|out| out.status.success());
    if !available {
        eprintln!("skip zip python check: pixi, zarr or xarray unavailable");
        return;
    }
    let out = Command::new("pixi")
        .current_dir(&root)
        .args(["run", "python", "-c", PYTHON_CHECK])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "python zip check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
