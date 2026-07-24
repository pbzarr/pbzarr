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

For now pbzarr imports from existing formats (d4, bigWig, and BED) rather than generating signal itself, but it does so fast, and once the data is in a pbz store the analysis speedups and disk savings make the conversion worth it. This is all still in active development, so expect rough edges and changes.

## Quickstart (Python)

Create an empty store, then materialize tracks by importing from a source file. The contig names and lengths come from the source header (d4 and bigWig), and the dtype is set by the format (d4 is `int32`, bigWig is `float32`), so there is nothing else to declare:

```python
import pbzarr

store = pbzarr.PbzStore.create("cohort.pbz")

# One source -> a 1D scalar track.
store.track("mean_depth").import_d4([("sample.d4",)])

# Several sources -> a 2D (position, sample) track, labelled in order.
store.track("depth").import_d4([("A.d4", "A"), ("B.d4", "B"), ("C.d4", "C")])

# bigWig imports the same way, into a float32 track.
store.track("signal").import_bigwig([("sample.bw",)])
```

BED import needs an explicit genome (`.fai` / chrom.sizes), since BED files carry no contig lengths. Import one named column per call, or every column in a single pass:

```python
# One named column across N bgzipped, tabix-indexed BEDs.
store.track("score").import_bed(
    [("a.bed.gz", "A"), ("b.bed.gz", "B")],
    column="score", dtype="float32", genome="hg38.fai",
)

# Many columns at once, one scalar track per column.
store.import_bed_multi(
    "calls.bed.gz",
    {"score": "float32", "qual": "int32", "pass": "bool"},
    genome="hg38.fai",
)
```

Combine many single-sample stores into one cohort store along the sample axis with `stack`. Each scalar track shared by all sources becomes a `(position, sample)` track; the labels default to each store's filename stem:

```python
cohort = pbzarr.stack(
    [("s1.pbz", "s1"), ("s2.pbz", "s2"), ("s3.pbz", "s3")],
    out="cohort.pbz",
    column_dim="sample",
)
```

### Read

Tracks are the read unit. A region query resolves 0-based, half-open coordinates to a slice and returns an `xr.DataArray`:

```python
store = pbzarr.PbzStore("cohort.pbz")
store.tracks()                                      # ['depth', 'signal', ...]

depth = store.track("depth")
depth.region("chr1:1000-2000")                      # (position, sample) DataArray
depth.region("chr1:1000-2000", column="A")          # one sample
depth.region(["chr1:0-500", "chr2:10-20"])          # gather several regions

# Reduce many intervals into a (region x column) matrix, featureCounts-style.
depth.region_reduced([("chr1", 0, 500), ("chr2", 10, 20)], reduce="mean")
```

Reads default to lazy/dask, aligned to the on-disk chunk grid; pass `chunks=None` for eager numpy (`store.track("depth", chunks=None)`), or open the whole store as a plain xarray `DataTree`:

```python
dt = pbzarr.open("cohort.pbz")            # eager
dt = pbzarr.open("cohort.pbz", chunks={}) # dask-backed
```

Since a pbz store is just a Zarr v3 store, anything that reads Zarr, including plain `zarr-python` and `xarray`, can read it directly.

## Format at a glance

- **Layout:** a flat, self-describing track layout. The store root is a bare `zarr_conventions` marker group; each track is its own group sized to ΣL (the sum of its contig lengths), with no per-contig subdivision.
- **Track group:** a `values` array (1D `(ΣL,)` for scalar tracks, 2D `(ΣL, n_columns)` for 2D tracks), an `offsets` prefix-sum index over `contigs`, the `contigs` name array, and, for 2D tracks, a column-label array named after the column dim.
- **Genome is per-track.** Each track owns its own `contigs`/`offsets` and `genome_checksum`; the store holds no genome, so two tracks in one store may cover different genomes.
- **Position is the first axis;** an optional column dim (default name `"column"`, often overridden to `"sample"`) is the second.
- **Coordinates:** 0-based, half-open.
- **Compression:** Blosc(zstd-5, byte-shuffle) on every `values` array.
- **File extension:** `.pbz`.

For the full design see [`docs/DESIGN.md`](docs/DESIGN.md).

## Links

- Rust API docs: [docs.rs/pbzarr](https://docs.rs/pbzarr)
- Design doc: [`docs/DESIGN.md`](docs/DESIGN.md)
- Motivating issues: [d4-format#82](https://github.com/38/d4-format/issues/82), [d4-format#64](https://github.com/38/d4-format/issues/64), [clam#25](https://github.com/cademirch/clam/issues/25)

## Development note

The per-base Zarr format that pbzarr standardizes was first prototyped by hand in [clam](https://github.com/cademirch/clam), where the initial concepts (the contig-major layout, cohort-shaped tracks, and the zarr/ndarray I/O path) were fleshed out before AI tooling was introduced. pbzarr lifts those concepts into a dedicated, spec-driven library.

From that point, development of pbzarr was heavily assisted by Claude (Anthropic), accelerating the library implementation, d4 import, tests, and documentation. The architecture, domain knowledge, and direction remain the author's own; Claude was used as an accelerant, not an author.
