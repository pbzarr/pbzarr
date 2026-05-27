# pbzarr

A Zarr v3 convention for storing per-base resolution genomic data — read depths, methylation, boolean masks, and other cohort-shaped per-base values. Built as an alternative to D4 and bigWig that compresses cleanly across samples and integrates with the xarray / zarr ecosystem.

This repo provides:

- **`pbzarr` (Rust crate)** — store layout, metadata, region I/O, d4 ingest. Delegates array storage and compression to [`zarrs`](https://crates.io/crates/zarrs).
- **`pbzarr` (Python wheel)** — PyO3 binding for `import_d4` plus pure-Python `create_store` / `create_track` over [`zarr-python`](https://github.com/zarr-developers/zarr-python). Read API is an xarray accessor.

The on-disk format is the same; both libraries write `.pbz` stores that the other can read.

## Install — Rust

```toml
[dependencies]
pbzarr = "0.1"
```

## Install — Python

```bash
pip install pbzarr            # once published on PyPI
# or from source: pixi run install-wheel
```

The Python wheel pulls in `zarr>=3`, `xarray>=2024.10`, `numpy>=2`.

## Quickstart — Python

```python
import pbzarr

# 1. Create the store
pbzarr.create_store(
    "out.pbz",
    contigs=["chr1", "chr2"],
    contig_lengths=[248_956_422, 242_193_529],
    coordinate_space="GRCh38",
)

# 2. Register a 1D scalar track
pbzarr.create_track("out.pbz", track="mask", dtype="bool")

# 2b. Or a 2D cohort track
pbzarr.create_track(
    "out.pbz",
    track="depth",
    dtype="uint32",
    columns=["A", "B", "C"],
    column_dim="sample",
)

# 3a. Bulk-ingest from d4 (PyO3 -> Rust)
pbzarr.import_d4(
    "out.pbz",
    track="depth",
    sources=[("/data/A.d4", "A"), ("/data/B.d4", "B"), ("/data/C.d4", "C")],
)

# 3b. Or write arbitrary numpy data via zarr-python
import zarr, numpy as np
g = zarr.open_group("out.pbz", mode="r+")
g["chr1/mask"][:] = np.random.rand(248_956_422) > 0.5
```

### Read with xarray

```python
import pbzarr

dt = pbzarr.open("out.pbz")                 # xr.DataTree
dt.pbz.tracks                                # ['depth', 'mask']
dt.pbz.region("chr1:1000-2000")              # xr.Dataset (one contig, sliced)
dt.pbz.region("chr1:1000-2000", track="depth")             # xr.DataArray
dt.pbz.region("chr1:1000-2000", track="depth", column="A") # 1D DataArray
```

The `.pbz` accessor on `xr.DataTree` is registered when you `import pbzarr`. Regions use 0-based, half-open coordinates (`chr1:1000-2000` is `[1000, 2000)`).

> **Note:** Python `create_store` / `create_track` consolidate metadata after each call. Stores written by the Rust crate don't consolidate yet, so `pbzarr.open(...)` will emit a benign `RuntimeWarning` for those; run `zarr.consolidate_metadata(path)` once to silence.

## Quickstart — Rust

```rust
use ndarray::Array2;
use pbzarr::io::Dtype;
use pbzarr::ingest::{import_d4, D4Source, ImportConfig};
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 248_956_422 },
        Contig { name: "chr2".into(), length: 242_193_529 },
    ])?;
    let mut store = PbzStore::create("out.pbz", genome, Some("GRCh38".into()))?;

    // 1D scalar track
    store.create_track("mask", TrackConfig::new(Dtype::Bool))?;

    // 2D cohort track; d4 ingest requires uint32
    store.create_track(
        "depth",
        TrackConfig::new(Dtype::U32)
            .columns(vec!["A".into(), "B".into(), "C".into()])
            .column_dim("sample"),
    )?;

    // Ingest from d4
    import_d4(
        &store,
        "depth",
        &[
            D4Source { path: "/data/A.d4".into(), sample_label: Some("A".into()) },
            D4Source { path: "/data/B.d4".into(), sample_label: Some("B".into()) },
            D4Source { path: "/data/C.d4".into(), sample_label: Some("C".into()) },
        ],
        ImportConfig::default(),
    )?;

    // Read a region
    let chr1 = store.genome().id("chr1").unwrap();
    let region = Region { contig: chr1, start: 1_000, end: 2_000 };
    let data = store.track("depth").unwrap().read_region::<u32>(&region)?;
    let arr2: Array2<u32> = data.into_dimensionality::<ndarray::Ix2>()?;
    let _ = arr2;
    Ok(())
}
```

## Format at a glance

- **Layout:** contig-major Zarr v3 store with `<contig>/<track>` arrays. Position is the first axis; an optional column dim (default name `"column"`, often overridden to `"sample"`) is the second.
- **Tracks:** 1D for scalar (e.g., masks), 2D for cohort (e.g., per-sample depths). Rank-faithful on disk.
- **Coordinates:** 0-based, half-open.
- **Compression:** Blosc(zstd-5, byte-shuffle) on every data array.
- **Coord arrays:** cohort tracks write per-contig 1D string arrays at `<contig>/<column_dim>` listing the column labels; xarray promotes them to coordinates automatically.

For the full design see [`docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md`](docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md).

## Status

v0 is **library + d4 ingest + Python wheel + xarray accessor**. Cross-language round-trip tests in both directions are green. CLI binary (`pbz`), additional ingest formats (bigWig, bedGraph, BED, BAM/CRAM), validation helpers, benchmarks, and a formal spec doc are deferred.

## Links

- Rust API docs: [docs.rs/pbzarr](https://docs.rs/pbzarr)
- Design doc: [`docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md`](docs/superpowers/specs/2026-05-25-pbz-v0-ship-design.md)
- Motivating issues: [d4-format#82](https://github.com/38/d4-format/issues/82), [d4-format#64](https://github.com/38/d4-format/issues/64), [clam#25](https://github.com/cademirch/clam/issues/25)
