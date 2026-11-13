# Upgrading to pbzarr 0.8

pbzarr 0.8 changes the store format, the command line, the Python API and the Rust crate. This page lists what you must change when you move from 0.7.

## Summary

- 0.8 writes and reads format 0.5 only. It cannot open a store made by 0.7 (format 0.4). Import your data again with 0.8.
- In `pbz`, the store is now the first argument, and every command takes `--track`. `-o/--output` is gone.
- `pbzarr.open` now returns one `xarray.Dataset`. The `.pbz` accessors are gone. Use the module functions `pbzarr.region` and `pbzarr.reduce_regions`.
- Several flags and keywords have new names. There are no aliases for the old names.
- New stores use a new default layout: inner chunks of 65,536 positions x 1 column, in shards of 1,024 chunks x 16 columns.
- BED value columns without data now hold the null (NaN for floats, the type maximum for integers), not 0.

## Stores

A 0.7 store has format 0.4. 0.8 refuses it with this error:

```text
This is a format 0.4 store. pbzarr 0.8 reads format 0.5 only. Use pbzarr 0.7, or import the data again.
```

There is no converter. Import the source files again with 0.8. If you must read old stores, keep pbzarr 0.7 in a separate environment.

Format 0.5 is the draft that format 1.0 will come from. Some rules can still change before 1.0, so a 1.0 reader may not open a 0.5 store.

What changed in the format:

- In format 0.4, each track was a group with a `values` array and its own contig and offset arrays.
- In format 0.5, the store is one zarr group, and each track is an array at the root. The first dimension of a track is `position` (or `bin_N` for binned tracks). The contigs are the root arrays `contig_id` and `contig_length`. Coordinates are 0-based and half-open.
- Every import, append and write ends by writing consolidated metadata at the root, so zarr-python and xarray open the store quickly.
- A format 0.5 store opens as an `xarray.Dataset` with `xr.open_zarr`, with no pbzarr code.

## Command line (`pbz`)

### Store and track arguments

| 0.7 | 0.8 |
|---|---|
| `pbz import d4 -o out.pbz --track depth a.d4 b.d4` | `pbz import d4 out.pbz a.d4 b.d4 --track depth` |
| `pbz view out.pbz depth` | `pbz view out.pbz --track depth` |
| `pbz stat out.pbz depth -r regions.bed` | `pbz stat out.pbz --track depth -R regions.bed` |

- Import creates the store from the source headers when the path does not exist. When the store exists, import adds the new track. A track name that is already in use is refused.
- `pbz import bed` needs `--genome` only when it creates the store.
- `-v/--verbose` is now a global flag. A new global flag `-q/--quiet` hides the summary line and the progress display.
- The import summary now goes to stderr, for every import: `imported N sources -> PATH (X positions, Y chunks written, Z skipped)`.

### Renamed flags

| 0.7 | 0.8 | Note |
|---|---|---|
| `--chunk-size N` | `--chunk N` | Positions per inner chunk. Default 65,536 (was 1,000,000). |
| `--shard-size N` | `--shard N` | The unit changed: it now counts inner chunks per shard, not positions. Default 1,024. |
| `--shard-column-size N` | `--shard-columns N` | Columns per shard. Default 16. |
| `-c/--column-dim` | `--dim` | No short form. |
| `--reference` (`import bam`) | `--fasta` | |
| `-r BED` (`stat`) | `-R/--regions BED` | `-r` now takes one region, `CHROM:START-END`. |

### Removed flags and commands

- `--column-chunk-size`: an inner chunk of a 2D track is now always 1 column wide. Use `--codecs` for another layout.
- `--dry-run`, `--in-flight`, `--scales` and `--progress` (`--no-progress` stays).
- `pbz scale`: multiscale pyramids are no longer made.

### Other changes

- `pbz view` and `pbz stat` take regions with `-r CHROM:START-END` (0-based, half-open, repeatable) and `-R BED`.
- `pbz import bed --emit-schema` now takes the BED file as its value: `--emit-schema FILE`. The schema has a fourth column, `fill`.
- `--exclude-flags` (`import bam`) also takes hex values and samtools flag names. The default is the same as before.
- `PATH[:LABEL]` sources: URLs are no longer split at `:`. Default labels drop compound extensions (`s1.bed.gz` gives `s1`). Paths in a `--file-list` file are resolved from the folder of that file.
- `pbz view` prints an integer null as NaN, and it leaves out rows where every column is null.
- New commands: `pbz append`, `pbz info` and `pbz validate`. New flag: `pbz import --zip`.

## Python (`pbzarr`)

The package needs Python 3.12 or later.

### Opening stores

- `pbzarr.open` always returns one `xarray.Dataset`. Each track is a data variable with the name of the track. There is no `values` variable, and there are no `DataTree` or collection nodes.
- The default dask chunks are now about 128 MiB each, made of whole inner chunks. Before, a dask chunk was one stored shard. Use `chunks=` to set them.
- `open` reads `.zip` stores (read-only).

### Renamed and removed functions

| 0.7 | 0.8 |
|---|---|
| `pbzarr.create_store(path)` | `pbzarr.create(path, reference={"chr1": 248956422, ...})` |
| `ds.pbz.region(query, column=...)` | `pbzarr.region(obj, query)` (no `column=`; select columns with xarray) |
| `ds.pbz.reduce_regions(intervals, reducer, **kwargs)` | `pbzarr.reduce_regions(obj, intervals, reducer, q=, by=, ddof=, skipna=)` |
| `ds.pbz.regions(...)`, `dt.pbz.regions(...)`, `dt.pbz.dataset(...)` | Removed. Use `pbzarr.region` or `pbzarr.reduce_regions`. |
| `RegionQuery.stop` | `RegionQuery.end` |

New: `pbzarr.write`, `pbzarr.interval_mask`, `pbzarr.Reference` and `pbzarr.__version__`.

### Import keywords

The changes apply to `import_d4`, `import_bigwig`, `import_bed` and `import_bam`.

| 0.7 | 0.8 |
|---|---|
| `column_dim=` | `dim=` |
| `workers=` | `threads=` |
| `chunk_size=` | `chunk=` |
| `shard_size=` (positions) | `shard=` (inner chunks per shard) |
| `shard_column_size=` | `shard_columns=` |
| `reference=` (`import_bam`) | `fasta=` |
| `column_chunk_size=`, `in_flight=`, `scales=` | Removed |
| `column_labels=` (`import_bam`) | Removed. Give `(path, label)` pairs as sources. |

- An import creates the store when it does not exist. You do not need `create_store` first.
- `import_bed` needs `genome=` only when it creates the store.
- `import_bigwig` and `import_bed` take `fill=` (a number or `"na"`).

### Import report

The dict that an import returns has these keys: `positions`, `chunks_written`, `chunks_skipped`, `bytes_written` and `wall_seconds`. The other keys (`contigs_written`, `tasks_completed`, `pieces`, the timing keys and others) are gone.

### Region results

- The coordinate `region_stop` is now `region_end`. `region_input_index` and `region_storage_index` are gone.
- `reduce_regions` returns rows in input order. Before, they were in storage order.
- Overlapping intervals are allowed. Before, they were refused.
- A DataFrame of intervals must have an `end` column. The `stop` alias is gone.
- An option that the reducer does not take raises `TypeError`.

### Errors

- A missing file raises `FileNotFoundError`.
- A bad argument, dtype, region or contig raises `ValueError`.
- Other failures raise `pbzarr.PbzError`.
- Ctrl-C stops an import and raises `KeyboardInterrupt`.

### Dependencies

- Python 3.12 or later (was 3.11).
- `flox` is no longer a dependency.
- New dependencies: `pandas` and `pcodec`.
- Higher minimum versions: `zarr>=3.2.1`, `xarray>=2025.12.0`, `numpy>=2.4.5`, `dask>=2026.8.0`.

## Rust crate (`pbzarr`)

- `PbzStore` and `Track` are replaced by `Dataset` (`open`, `open_read_only`, `open_zip`, `create`, `create_zip`, `read_region`, `validate` and more). `Measurement::root(&dataset, name)` gives the handle of one track.
- `Genome` is replaced by `Reference`. `Genome::resolve` is now `Reference::region`.
- The `genome`, `scale`, `store` and `track` modules are removed. `scale`, `ScaleConfig` and the estimate types are removed.
- `TrackConfig`:
  - `chunk_size` is now `chunk`.
  - `shard_size` is now `shard`, in inner chunks per shard. `unsharded()` makes a track with no shards.
  - `shard_column_size` is now `shard_columns`.
  - `column_chunk_size` is removed.
  - `fill(Fill)` sets the fill, and `Fill::Na` declares the null as the fill value.
- `ImportBuilder::into_track` and `into_tracks` are now `into_dataset`. `options(PipelineOptions)` is now `config(Config)`. `with_cost_model`, `with_tap` and `estimate()` are removed.
- `import::Config` uses `chunk`, `shard` and `shard_columns`. `in_flight_spans`, `handle_budget` and `scales` are removed.
- `Report` has the fields `positions`, `chunks_written`, `chunks_skipped`, `bytes_written` and `wall_seconds`.
- `Dtype` has `I64` and `U64`.

## New defaults

### Layout

| | 0.7 | 0.8 |
|---|---|---|
| Inner chunk | 1,000,000 positions x 16 columns | 65,536 positions x 1 column |
| Shard | none | 1,024 inner chunks (67,108,864 positions) x 16 columns |
| Codec | Blosc zstd level 5, shuffle | the same |

- A shard is never larger than the genome or the track width.
- In our tests with 32 whole-genome samples, the new layout made stores one third smaller (44.4 GB to 29.9 GB) and made 104 files in place of about 6,400.
- `pbzarr.write` gives the same layout to new variables whose first dimension is `position`, unless `encoding=` sets the layout.

### Fill of BED value columns

In 0.7, a BED value column held 0 at positions with no data. In 0.8 it holds the null of its dtype: NaN for floats and the type maximum for integers. The null is the `_FillValue` of the track, so xarray reads it as NaN, and an integer track then has a float dtype in xarray. A BED value equal to the integer maximum is refused.

Bool columns and BED3 masks still hold `false`. To get the 0.7 behaviour, use `--fill 0` (CLI) or `fill=0` (Python).

### Append

`pbz append` cannot add columns to a track whose shards hold more than one column, and the new default is 16 columns per shard. To append to a track later, import it with `--shard-columns 1`.
