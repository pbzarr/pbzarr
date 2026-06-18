<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pbzarr-grid-dark.svg">
  <img alt="pbzarr" src="docs/assets/pbzarr-grid-light.svg">
</picture>

# pbzarr

pbzarr stores per-base genomic data (read depths, methylation, boolean masks, and other per-position values) in Zarr v3. It is a layout and metadata convention on top of Zarr, not a new binary format: array storage, chunking, and compression are handled by the underlying Zarr library.

The reason to use Zarr this way is the cohort case. One-file-per-sample formats like D4 and bigWig store each sample on its own, so they do not compress across samples and any cross-sample computation becomes a loop over files. pbzarr keeps many samples in a single chunked `(position, sample)` array, so samples compress together and per-position math across a cohort is one vectorized read. Single-sample stores are still first-class and read back as xarray.

This repo provides:

- **`pbzarr` (Rust crate):** store layout, metadata, and region I/O. Delegates array storage and compression to [`zarrs`](https://crates.io/crates/zarrs). This is the only crate published to crates.io.
- **`pbzarr-readers` (Rust crate, unpublished):** import readers for d4 and bigWig. Kept separate so the core crate carries no git dependencies and stays publishable. The readers are reachable from the Python wheel or when building from this repo, not from the crates.io `pbzarr` crate.
- **`pbzarr` (Python wheel):** a `PbzStore` class over [`zarr-python`](https://github.com/zarr-developers/zarr-python) for creating stores and tracks, PyO3 bindings for d4 and bigWig import, and an xarray-based read path.

Both libraries write the same `.pbz` layout, and either can read what the other writes.

## Install (Rust)

```toml
[dependencies]
pbzarr = "0.2"
```

## Install (Python)

```bash
pip install pbzarr            # once published on PyPI
# or from source, from this repo:
pixi run install-wheel
```

The Python wheel pulls in `zarr>=3`, `xarray`, `numpy>=2`, and `dask`.

## Quickstart (Python)

The fastest way to build a store is straight from a d4 or bigWig file. The contig names and lengths are read from the source header, and the dtype is set by the format (d4 is `int32`, bigWig is `float32`), so there is nothing else to declare:

```python
import pbzarr

# Single sample: a 1D scalar track.
store = pbzarr.PbzStore.from_d4("sample.pbz", "sample.d4", track="depth")

# A cohort: pass {label: path}. The keys become the column labels, in order,
# and the result is a 2D (position, sample) track. Every source must map to
# the same reference; a mismatched contig set raises.
store = pbzarr.PbzStore.from_d4(
    "cohort.pbz",
    {"A": "A.d4", "B": "B.d4", "C": "C.d4"},
    track="depth",
    column_dim="sample",
)

# bigWig is the same call, into a float32 track.
store = pbzarr.PbzStore.from_bigwig("signal.pbz", "sample.bw", track="signal")
```

Or build the store by hand when you need full control over the layout:

```python
import numpy as np
import zarr

store = pbzarr.PbzStore.create(
    "out.pbz",
    contigs=["chr1", "chr2"],
    contig_lengths=[248_956_422, 242_193_529],
    coordinate_space="GRCh38",
)

# A 1D scalar track and a 2D cohort track.
store.create_track("mask", dtype="bool")
store.create_track("depth", dtype="int32", columns=["A", "B", "C"], column_dim="sample")

# Bulk-import the cohort track from d4 (PyO3 -> Rust). Use import_bigwig for bigWig.
store.import_d4("depth", sources=[("A.d4", "A"), ("B.d4", "B"), ("C.d4", "C")])

# Or write arbitrary numpy data through zarr-python directly.
g = zarr.open_group("out.pbz", mode="r+")
g["chr1/mask"][:] = np.random.rand(248_956_422) > 0.5
```

### Read with xarray

```python
store = pbzarr.PbzStore("out.pbz")
store.tracks                                       # ['depth', 'mask']
store.region("chr1:1000-2000", track="depth")      # xr.DataArray
store.region("chr1:1000-2000", track="depth", column="A")  # one sample

# Or open the whole store as an xarray DataTree via the .pbz accessor:
dt = pbzarr.open("out.pbz")
dt.pbz.region("chr1:1000-2000", track="depth")
```

The `.pbz` accessor on `xr.DataTree` is registered when you `import pbzarr`. Regions use 0-based, half-open coordinates, so `chr1:1000-2000` is `[1000, 2000)`.

> **Note:** Python `PbzStore.create` / `create_track` consolidate metadata after each call. Stores written by the Rust crate do not consolidate yet, so `pbzarr.open(...)` emits a benign `RuntimeWarning` for those; run `zarr.consolidate_metadata(path)` once to silence it.

## Quickstart (Rust)

```rust
use ndarray::Array2;
use pbzarr::io::Dtype;
use pbzarr::import::Config;
use pbzarr::{Contig, Genome, PbzStore, Region, TrackConfig};
// Import readers live in the unpublished pbzarr-readers crate (it carries a git
// dependency on d4); depend on this repo by path or git to use them.
use pbzarr_readers::d4::{from_d4, D4Source};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let genome = Genome::new(vec![
        Contig { name: "chr1".into(), length: 248_956_422 },
        Contig { name: "chr2".into(), length: 242_193_529 },
    ])?;
    let mut store = PbzStore::create("out.pbz", genome, Some("GRCh38".into()))?;

    // 1D scalar track
    store.create_track("mask", TrackConfig::new(Dtype::Bool))?;

    // 2D cohort track; d4 import requires int32
    store.create_track(
        "depth",
        TrackConfig::new(Dtype::I32)
            .columns(vec!["A".into(), "B".into(), "C".into()])
            .column_dim("sample"),
    )?;

    // Import from d4
    from_d4(
        &store,
        "depth",
        &[
            D4Source { path: "/data/A.d4".into(), sample_label: Some("A".into()) },
            D4Source { path: "/data/B.d4".into(), sample_label: Some("B".into()) },
            D4Source { path: "/data/C.d4".into(), sample_label: Some("C".into()) },
        ],
        Config::default(),
    )?;

    // Read a region
    let chr1 = store.genome().id("chr1").unwrap();
    let region = Region { contig: chr1, start: 1_000, end: 2_000 };
    let data = store.track("depth").unwrap().read_region::<i32>(&region)?;
    let arr2: Array2<i32> = data.into_dimensionality::<ndarray::Ix2>()?;
    let _ = arr2;
    Ok(())
}
```

bigWig import is the mirror image: `pbzarr_readers::bigwig::{from_bigwig, BigWigSource}` into a `float32` track.

## Format at a glance

- **Layout:** contig-major Zarr v3 store with `<contig>/<track>` arrays. Position is the first axis; an optional column dim (default name `"column"`, often overridden to `"sample"`) is the second.
- **Tracks:** 1D for scalar values (e.g., masks), 2D for cohort values (e.g., per-sample depths). Rank-faithful on disk.
- **Coordinates:** 0-based, half-open.
- **Compression:** Blosc(zstd-5, byte-shuffle) on every data array.
- **Coord arrays:** cohort tracks write per-contig 1D string arrays at `<contig>/<column_dim>` listing the column labels; xarray promotes them to coordinates automatically.

For the full design see [`docs/DESIGN.md`](docs/DESIGN.md).

## Links

- Rust API docs: [docs.rs/pbzarr](https://docs.rs/pbzarr)
- Design doc: [`docs/DESIGN.md`](docs/DESIGN.md)
- Motivating issues: [d4-format#82](https://github.com/38/d4-format/issues/82), [d4-format#64](https://github.com/38/d4-format/issues/64), [clam#25](https://github.com/cademirch/clam/issues/25)

## Development note

The per-base Zarr format that pbzarr standardizes was first prototyped by hand in [clam](https://github.com/cademirch/clam), where the initial concepts (the contig-major layout, cohort-shaped tracks, and the zarr/ndarray I/O path) were fleshed out before AI tooling was introduced. pbzarr lifts those concepts into a dedicated, spec-driven library.

From that point, development of pbzarr was heavily assisted by Claude (Anthropic), accelerating the library implementation, d4 import, tests, and documentation. The architecture, domain knowledge, and direction remain the author's own; Claude was used as an accelerant, not an author.
