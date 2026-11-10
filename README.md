<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pbzarr-grid-dark.svg">
  <img alt="pbzarr" src="docs/assets/pbzarr-grid-light.svg">
</picture>

# pbzarr

## What is this?

pbzarr is a library for storing and analyzing genomic measurements at base-pair resolution. It stores measurements in two-dimensional arrays: one dimension for genomic position, and another for samples or other labels. You can run calculations across either dimension or both.

pbzarr uses [Zarr](https://zarr.dev/), an existing format for compressed arrays. It adds conventions for genomic coordinates and column labels, plus tools to import and analyze the data.

## Motivation

Most bioinformatics files (BAM, BED, bigWig) hold data for one sample. And this makes sense, because most primary bionformatics analyses (i.e alignment) are sample independent. But many downstream analyses need comparisons across samples, over the whole genome or in regions of interest. That means opening each sample's files, reading the relevant data, performing computations, and storing the result.

Often these analyses are exploratory. We run them several times as we decide which samples to include, which regions to look at, or where to set a threshold. With a few regions or a small sample set, this is not that big of a deal. But as cohorts and region sets grow, more work goes into reading the files and bringing the measurements together. During exploratory analysis, we can repeat that work every time we ask a slightly different question, even though the underlying measurements have not changed.

pbzarr builds on the idea that these analyses often need the same two-dimensional array: one dimension for genomic position, and another for samples. pbzarr builds this array for the whole genome once and stores it in compressed chunks using [Zarr](https://zarr.dev/). Each analysis can then read the regions and samples it needs, without rebuilding the array from individual sample files.

This also has a few useful side effects:

- [Xarray](https://docs.xarray.dev/en/stable/user-guide/data-structures.html) lets us attach genomic positions, sample names, and other labels to the dimensions, and organize different measurements together.
- [Dask](https://docs.dask.org/en/stable/array.html) lets us run calculations in parallel, on one machine or across a cluster.
- [Zarr](https://zarr.readthedocs.io/en/stable/user-guide/storage/) supports local and cloud storage. We can read the chunks we need from the cloud without downloading the whole dataset.

## Data model

pbzarr uses the [dataset model of xarray](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset). A `.pbz` store opens as one xarray Dataset. Each track is one variable, and all tracks share the store's reference genome and labels.

A track has at most two dimensions: genomic position and one labeled column axis, such as `sample`. The column axis is homogeneous, so every column shares the track's data type, fill value, and compression settings. When values need different data types or settings, each kind goes into its own track. For example, depth, mapping quality, and a mask become three tracks in one store.

The store fixes its reference (contig names and lengths) when it is created. Every track covers that whole reference. See [format](#format) for more details.

## Python examples

### Import d4 files

```python
import pbzarr

pbzarr.import_d4(
    "depth.pbz",
    "depth",
    [("brain.d4", "brain"), ("liver.d4", "liver")],
    dim="sample",
)
```

This creates the store `depth.pbz` and a `depth` track with one row per genomic position and one column per sample. The import takes the store's reference from the source headers. All sources must have the same contig names, lengths, and order.

When the store exists, the import adds the track to it. The sources must then match the store's reference. The import refuses a track name that the store already uses. The CLI imports follow the same rule.

### Read a region

```python
ds = pbzarr.open("depth.pbz")

window = pbzarr.region(ds["depth"], "chr1:100000-101000")
brain = window.sel(sample="brain")

values = window.compute()
```

`pbzarr.open` returns an ordinary `xarray.Dataset`, and `ds["depth"]` is a lazy `xarray.DataArray` of the whole genome. `pbzarr.region` gives a view of one locus. The view labels each base as `"contig:pos"`, and its numeric `pos` coordinate holds the contig-local positions. pbz uses zero-based, half-open coordinates.

### Calculate statistics over regions

```python
peaks = [
    ("chr1", 100000, 100200),
    ("chr1", 150000, 150300),
    ("chr2", 200000, 200150),
]

peak_means = pbzarr.reduce_regions(ds, peaks, "mean")
result = peak_means.compute()
```

The result has one mean for each peak and sample. Other reductions include `sum`, `min`, `max`, `count`, `std`, `var`, `median`, and `quantile`. Regions can also be a list of strings such as `"chr1:100000-100200"`, or a dataframe with `chrom`, `start`, and `end` columns.

Dask keeps reads and calculations lazy until `compute()`. The computed result must still fit in memory.

### Mask positions inside regions

```python
mask = pbzarr.interval_mask(ds, peaks)
peak_depth = ds["depth"].where(mask)
```

`pbzarr.interval_mask` gives a lazy bool array over `position`. A value is True inside a region and False outside it. It takes the same region forms as `reduce_regions`, and the regions can overlap.

### Write a derived track

```python
ds = pbzarr.open("depth.pbz")
ds = pbzarr.write(ds, {"over_10": (ds["depth"] >= 10).mean("sample")})
```

A derived track must cover the whole reference on `position`, or have no genomic dimension. The write drops 0-d coordinates, for example `quantile` after `quantile(0.5)`. `pbzarr.write` takes a path, a URL, or a Dataset from `pbzarr.open`. It adds the track to the store, with the default Zarr codecs unless `encoding=` says otherwise, and returns the store opened again. It does not change the Dataset that you give it.

### Import other files

```python
pbzarr.import_bigwig("signals.pbz", "signal", "sample.bw")

pbzarr.import_bed(
    "scores.pbz",
    "sites.bed.gz",
    genome="genome.fai",
    track="score",
    column="score",
    dtype="float32",
)
```

BED does not record chromosome lengths. To create a store, a BED import needs a `.fai` or chromosome-sizes file (`genome=`, or `--genome` in the CLI). An import into an existing store takes the lengths from the store.

`pbzarr.create(path, reference=...)` makes an empty store for `pbzarr.write`. Imports do not need it.

## CLI examples

Import a bigWig file into a `signal` track:

```sh
pbz import bigwig -o signal.pbz --track signal sample.bw
```

The import reads chromosome lengths from the bigWig header and stores values as `float32`. Uncovered bases contain `NaN`.

Import several files as labeled columns:

```sh
pbz import bigwig -o cohort.pbz --track signal --dim sample brain.bw:brain liver.bw:liver
```

All sources must describe the same genome. Without explicit labels, the import uses each file's stem as its column label.

The command accepts the same import options as `pbz import d4`. These include `--file-list`, `--threads`, chunk and shard sizes, `--codecs`, and `--no-progress`.

Add samples to an existing cohort track:

```sh
pbz append bigwig -o cohort.pbz --track signal heart.bw:heart
```

Every variable over the appended dimension grows. Tracks that receive no data read their fill value in the new columns.

Read the sources from a list file:

```sh
pbz import bigwig -o cohort.pbz --track signal --file-list sources.tsv
```

The source list contains one path per line, optionally followed by a tab and a label. If any source has a label, all sources must have labels.

## Format

A `.pbz` store is one Zarr v3 group. Tracks and coordinates are arrays directly under the root.

```text
cohort.pbz/
├── zarr.json        root attributes (zarr_conventions, pbz:version), metadata copy
├── contig_id        contig names     (contig)
├── contig_length    contig lengths   (contig)
├── sample           column labels
├── depth            (position, sample)
└── over_10          (position,)
```

- The reference is the two root arrays `contig_id` and `contig_length`, over the dimension `contig`. The names are fixed. Their order defines the genome: the `position` dimension is all contigs concatenated in that order. An optional `contig_md5` array can hold a checksum for each contig.
- Readers find arrays by name and dimension names. A track is an array whose first dimension is `position` or `bin_N`. The `coordinates` attribute that pbzarr writes is only for xarray; a store without it opens the same way.
- Offsets are not stored. The first row of contig `i` is the sum of the lengths before it, so the rows for `chr2:10-20` are that sum plus 10 up to that sum plus 20. All coordinates are zero-based and half-open.
- A column axis such as `sample` has a label array of the same name. Tracks that share it share its labels.
- Binned tracks use a `bin_N` dimension. Bins restart at each contig, so a contig's last bin can be short.
- The root `zarr.json` holds a consolidated copy of the metadata of every array. Each import, append, and `pbzarr.write` replaces it as its last step, so a reader that cannot list the store can still find every array.

Every track declares one data type and fill value. Imported tracks are chunked and compressed with Blosc (zstd with byte shuffle) by default; imports can override the codec pipeline and chunk grid with standard Zarr v3 metadata (`--codecs` on the CLI, `codecs=` in Python). A store opens with plain xarray (`xr.open_zarr`), with no pbzarr code.

pbzarr reads and writes format 0.5 only. It refuses a format 0.4 store, which used one group per track. To read such a store, use pbzarr 0.7, or import the data again.

See the [format specification](spec/README.md) for the full rules and the [design document](docs/DESIGN.md) for the Rust API.

## Status

pbzarr is under active development and has not reached version 1.0. The API and file rules can change. Remote writes and variable chunk sizes are planned.

## Links

- [Rust API](https://docs.rs/pbzarr)
- [Design](docs/DESIGN.md)
- [d4](https://github.com/38/d4-format)
- [Motivating issues](https://github.com/38/d4-format/issues/82): mask generation, [cross-sample compression](https://github.com/38/d4-format/issues/64), and [per-base sample statistics](https://github.com/cademirch/clam/issues/25)

## Development note

The per-base Zarr format that pbzarr standardizes was first prototyped by hand in [clam](https://github.com/cademirch/clam), where the initial concepts (the contig-major layout, cohort-shaped tracks, and the zarr/ndarray I/O path) were fleshed out before AI tooling was introduced. pbzarr lifts those concepts into a dedicated, spec-driven library.

From that point, development of pbzarr was heavily assisted by Claude (Anthropic), accelerating the library implementation, d4 import, tests, and documentation. The architecture, domain knowledge, and direction remain the author's own; Claude was used as an accelerant, not an author.
