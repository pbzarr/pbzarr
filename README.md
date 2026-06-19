<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pbzarr-grid-dark.svg">
  <img alt="pbzarr" src="docs/assets/pbzarr-grid-light.svg">
</picture>

# pbzarr

pbzarr stores per-base genomic data (read depths, methylation, boolean masks, and other per-position values) in Zarr v3. It builds on the recent work in the Python array ecosystem: [Zarr](https://zarr.dev), [Xarray](https://xarray.dev), and [Dask](https://www.dask.org). pbzarr has a Rust and Python API for reading and writing pbz stores, and since a pbz store is just a Zarr store, anything that reads Zarr can read pbzarr.

A store holds one or more tracks. A track is usually a single quantitative signal or metric measured across the whole genome, one value per base. pbzarr keeps that idea but generalizes it: a track is an N-dimensional array indexed by genomic position along its first axis. A 1D track is the familiar one-value-per-base signal; add a second axis and you store many values per base, which is what makes it natural to keep multiple metrics or a whole cohort of samples in a single track.

A few things you get from this:

- Many values per base live in one array, so Zarr compresses the redundancy across that second axis, not just the runs within a single column.
- Analysis stays vectorized. Sums, means, and masks run across the whole array at once instead of looping over separate files in Python.
- Xarray and Dask work directly on the store, so there's no separate conversion step before you can label, slice, or parallelize.

pbzarr can be thought of as the spiritual successor of [d4](https://github.com/38/d4-format/tree/master). d4 improved on bigWig with compression, better throughput, multi-track files, and a cleaner API. pbzarr pushes on all of those, mostly by leaning on the work behind Zarr, Xarray, and Dask rather than reinventing it.

For now pbzarr imports from existing formats (d4 and bigWig) rather than generating signal itself, but it does so fast, and once the data is in a pbz store the analysis speedups and disk savings make the conversion worth it. This is all still in active development, so expect rough edges and changes.

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
