<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pbzarr-grid-dark.svg">
  <img alt="pbzarr" src="docs/assets/pbzarr-grid-light.svg">
</picture>

# pbzarr

## Synopsis

pbzarr stores per-base genomic data in Zarr v3. A `.pbz` collection can hold depth, signal, masks, and other tracks for one sample or many samples.

pbzarr supports:

- fast reads over genomic regions
- one value or many labeled values at each base
- compression across samples or other columns
- parallel calculations with xarray and Dask
- d4, bigWig, and BED imports

The project provides Rust and Python libraries. A command-line tool is planned.

## Motivation

Per-base data often lives in one file per sample. Analysis then spends time opening files and joining results in Python.

pbzarr stores related values in one array. This improves compression and lets array tools calculate across positions and samples at once.

## Python examples

### Import d4 files

```python
import pbzarr

pbzarr.create_store("depth.pbz")
pbzarr.import_d4(
    "depth.pbz",
    "depth",
    [("brain.d4", "brain"), ("liver.d4", "liver")],
    column_dim="sample",
)
```

This creates a `depth` track with one row per genomic position and one column per sample.

### Read a region

```python
depth = pbzarr.open("depth.pbz/depth")

window = depth.pbz.region("chr1:100000-101000")
brain = depth.pbz.region("chr1:100000-101000", column="brain")

values = window.compute()
```

pbz uses zero-based, half-open coordinates.

### Calculate statistics over regions

```python
peaks = [
    ("chr1", 100000, 100200),
    ("chr1", 150000, 150300),
    ("chr2", 200000, 200150),
]

peak_values = depth.pbz.regions(peaks)
peak_means = peak_values.pbz.reduce("mean")
result = peak_means.compute()
```

The result has one mean for each peak and sample. Other reductions include `sum`, `min`, `max`, `count`, `std`, `var`, `median`, and `quantile`.

Dask keeps reads and calculations lazy until `compute()`. The computed result must still fit in memory.

### Open a collection

Opening a track returns an `xarray.Dataset`. Opening a collection returns an `xarray.DataTree` whose children are tracks.

```python
study = pbzarr.open("study.pbz")

depth = study["depth"].to_dataset()
mask = study["mask"].to_dataset()
```

Each track records its own genome. Tracks with matching genomes can be used together.

### Import other files

```python
pbzarr.create_store("signals.pbz")
pbzarr.import_bigwig("signals.pbz", "signal", "sample.bw")

pbzarr.create_store("scores.pbz")
pbzarr.import_bed(
    "scores.pbz",
    "score",
    "sites.bed.gz",
    column="score",
    dtype="float32",
    genome="genome.fai",
)
```

BED imports need a `.fai` or chromosome-sizes file because BED does not record chromosome lengths.

## Format

A `.pbz` collection is a Zarr v3 directory. Each child directory is a track.

```text
study.pbz/
├── zarr.json
├── depth/
│   ├── zarr.json
│   ├── values
│   ├── offsets
│   ├── contigs
│   └── sample
└── mask/
    ├── zarr.json
    ├── values
    ├── offsets
    └── contigs
```

`values` holds the data. `contigs` and `offsets` map each chromosome to its rows in `values`. A track with columns also stores their labels.

See the [design document](docs/DESIGN.md) for the full file rules and Rust API.

## Status

pbzarr is under active development and has not reached version 1.0. The API and file rules may change. Remote writes and variable chunk sizes are planned.

## Links

- [Rust API](https://docs.rs/pbzarr)
- [Design](docs/DESIGN.md)
- [d4](https://github.com/38/d4-format)
- [Motivating issues](https://github.com/38/d4-format/issues/82): mask generation, [cross-sample compression](https://github.com/38/d4-format/issues/64), and [per-base sample statistics](https://github.com/cademirch/clam/issues/25)
