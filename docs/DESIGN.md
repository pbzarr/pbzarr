# PBZ design

PBZ is a small Zarr v3 convention for dense, per-base genomic data. The format supplies genomic interpretation and a flat track layout; Zarr supplies storage and codecs; xarray and Dask supply the Python read and analysis model.

pbzarr reads and writes format 0.5 (one root Dataset) only; see the [format 0.5 specification](../spec/README.md). Earlier contig-major, per-track, and wrapper-oriented designs are historical only.

## Development reader

The additive Rust `pbzarr::Dataset` reads the
[format `0.5` root-Dataset schema](../spec/README.md).
It catalogs direct root arrays, validates reference and dimension coordinates,
and reads typed rectangular subsets. `read_raw` preserves physical values and
missing sentinels; `read_bool` decodes the supported Boolean encodings.
The module caches metadata and reference geometry, so callers must reopen after
external changes. It does not provide a concurrent-mutation snapshot.

CLI imports and appends write format `0.5`, and `pbz view` and `pbz stat`
read it. Python opens a `0.5` store as an `xr.Dataset` with `pbzarr.open`.
A format `0.4` store used one group per track. Every reader and writer, in
Rust, the CLI, and Python, refuses it with the same message. In Rust, the
error is `PbzError::LegacyFormat`.

The root `zarr.json` of a `0.5` store holds a consolidated metadata copy in
the zarr-python inline form. Every import, append, and Python write replaces
this copy as its last step. On a filesystem store, the Rust writer writes the
new root to a temporary file and renames it over `zarr.json`. Until that step,
the old copy shows the store as it was before the Rust write. A Python write
removes the copy while `to_zarr` runs. `Dataset::open` lists the store and does
not use the copy.

## Goals

PBZ makes per-base depth, signal, masks, and similar position-indexed data portable across Rust, Python, and ordinary Zarr tooling. It aims to preserve rank, keep genomic geometry explicit, and make xarray the normal Python read surface.

The format serves scalar signals as well as wide column tracks. A column axis can represent samples, strands, methylation contexts, or another declared label set; “cohort” is only the sample-specific case.

PBZ does not define a new container, codec, coordinate conversion layer, or query language. Coordinates are 0-based and half-open. Callers convert external 1-based formats at their own boundaries.

## Data model

A store holds one reference: ordered `(contig name, length)` pairs. Every track covers the whole reference, and the `position` dimension is all contigs concatenated in reference order.

A scalar track is a root array shaped `(position,)`. A 2D track is shaped `(position, column)`, where `column` is the declared dimension name. The labels of that dimension are a string array of the same name at the root, and tracks that share the dimension share its labels.

Every track declares a fill value: 0 for integer tracks, NaN for float tracks, false for bool tracks. The d4, BED, and BAM/CRAM importers fill uncovered positions with 0. A bigWig stores no value at an uncovered base, so bigWig tracks import as float32 with a NaN fill. xarray reductions skip NaN by default, so uncovered bases do not bias region statistics.

## Chunks and shards

By default, every imported track uses Blosc with zstd level 5 and byte shuffle. Scalar tracks are chunked by position; 2D tracks are chunked by position and column so compression can exploit similarity across columns. Import entry points accept an explicit Zarr v3 codec spec (`--codecs` on the CLI, `codecs=` in Python) that replaces this default: a standard `codecs` list, with sharding expressed as a `sharding_indexed` codec and an optional `chunk_grid` member for the outer grid.

Sharding is optional and has an outer position-first shard grid with full declared column width by default. The normal Zarr chunk shape is retained as the shard’s subchunk shape. Sharded writes append encoded subchunks through Zarr partial encoding instead of rewriting whole shards.

One import engine drives every format. A chunk is one inner position chunk, and it is the unit the engine writes. A span is a run of consecutive chunks, and it is the window one reader decodes in one pass. By default the engine sizes the span automatically: up to 4 million positions, and fewer when that leaves the workers short of pieces. A piece is one reader and one span. Each reader decodes each span once and fills every schema field in that one pass. Buffers still close and write per chunk, so the engine still skips an uncovered chunk inside a covered span. The engine pre-fills every buffer with the destination track’s declared fill value.

The worker that completes a buffer writes it. On an unsharded track, the buffer is one whole chunk. On a sharded track, the worker compresses the buffer and appends it to its shard exactly once, and a per-shard lock serializes the appends. The engine never writes a chunk that no reader covers. An absent chunk reads back as the fill value.

## Python API

The public namespace is `Reference`, `PbzError`, `open`, `create`, `write`, `RegionQuery`, `parse_region`, `reduce_regions`, `interval_mask`, `region`, `import_d4`, `import_bigwig`, `import_bed`, and `import_bam`. There is no store handle. Python reads through zarr-python and xarray; Rust is used only for imports.

`pbzarr.open(source, *, tracks=None, chunks=..., storage_options=None)` returns the root of a format 0.5 store as one `xr.Dataset`. It checks the pbz convention entry and version, and refuses a format 0.4 store. It reads `pbz:version` "0.5" only and ignores `pbz:` attributes that it does not know. It reads the store metadata from the consolidated copy in the root `zarr.json` when the store has one, and lists the store when it does not. `pbzarr.write` refuses a store of another version. `tracks=` narrows the data variables and keeps the reference and label coordinates. By default each track is a Dask array chunked by its stored shard (or chunk) shape; `chunks=None` gives lazy NumPy-backed arrays. The Dataset is a snapshot of the store at open time. A change in memory does not reach the store, and `close` and the context manager are those of `xr.Dataset`.

Every `xr.DataArray` and `xr.Dataset` from `open` carries a 0-d coordinate named `reference`. Its value is a short digest of the contig names and lengths. Its index, `ReferenceIndex`, holds the `Reference`. The coordinate has no dimension, so it stays through arithmetic, reductions, `concat`, `merge`, and `pickle`. When two objects with different references meet, xarray raises `AlignmentError`. `Reference.from_xarray(obj)` gives the reference back. The coordinate exists in memory only; no store holds it.

`Reference` is a mapping of contig name to length. `rows(query, dim="position")` maps one region to a slice of flat rows with the same rule as the Rust reader; on `bin_N` it gives the bins that intersect the region. `resolve(query)` gives the same region with contig-local `start` and `end` filled in: an open start is 0, and an open end is the contig end. Empty and out-of-range regions are refused. Select one locus with `pbzarr.region(ds["depth"], "chr1:100-200")`. `sel(position=...)` is reserved: with no genomic index, xarray treats it as row positions, and its meaning changes when the index arrives.

`pbzarr.region(obj, query, *, reference=None)` gives a view of one locus. `query` is a string such as `"chr1:100-200"` or `"chr1"`, a `RegionQuery`, or a `(contig, start, end)` tuple. `obj` is a whole-genome `xr.DataArray` or `xr.Dataset`, and the view has the same type. The reference and the full-length rule are the same as for `reduce_regions`. The function slices `position` and each `bin_N` dimension that is present through `Reference.rows`, so a Dataset view stays consistent. Variables with no genomic dimension pass through unchanged. The view carries the 0-d coordinates `region_contig`, `region_start`, and `region_end`. When `position` is present, the view also carries a numeric `pos` coordinate and the labels `"contig:pos"` as an indexed `position` coordinate. These positions are 0-based and contig-local, as `region_start` is. A `LocusIndex` on `position` holds the contig name and a range of positions. The labels and `pos` are made from it only when a caller reads the values, so a view of a whole contig adds no per-base memory. Two views of one contig align by label. Views of two contigs raise `AlignmentError` when they align, and `concat` of them keeps the labels in a plain index. `sel` takes exact labels, lists of labels, and label slices that include the end. The view carries the `reference` coordinate of the reference that was used. `region` and `reduce_regions` refuse a view that has a `position` dimension, because of its index, and a view that is shorter than the genome.

`pbzarr.reduce_regions(obj, intervals, reducer, /, *, reference=None, **kwargs)` reduces each complete interval along `position`, in one pass. `obj` is an `xr.DataArray` or an `xr.Dataset`, and the result has the same type, with a `region` dimension first. The reference comes from the `reference` coordinate. `reference=` overrides it, or gives it when the input has no coordinate, for example a Dataset from `xr.open_zarr`. Each genomic dimension must have its full length; a sliced input is refused. `position` can be at any place in the dimensions. A track whose first dimension is not `position` is transposed first, and then it does not use the fast path. On a Dataset, the variables with `position` are reduced. Other variables, `bin_N` variables included, and the coordinates that are not on `position` pass through unchanged. Coordinates on `position` are dropped. The result carries the `reference` coordinate of the reference that was used, and it has no Dataset attributes. A track may have at most one dimension in addition to `position`. Intervals may be a dataframe, three arrays, rows, or a list of region strings. A dataframe needs one `contig`, `chrom`, or `#chrom` column, a `start` column, and an `end` column. Supported reducers are `mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `median`, `quantile`, and `summit` (which requires a `by=` variable). The function does not detect a full-length reorder of `position`, such as `roll` or `sortby`.

`pbzarr.interval_mask(obj, intervals, /, *, reference=None)` gives a bool `xr.DataArray` over `position`. A value is True inside any interval and False outside. `obj` is a whole-genome `xr.DataArray` or `xr.Dataset` with a `position` dimension. The intervals take the same forms as for `reduce_regions`, and they can overlap. The reference and the full-length rule are the same as for `reduce_regions`. The mask is a Dask array with the `position` chunks of `obj`, or with the Dask default chunks when `obj` is not chunked. Each chunk is computed from the intervals when it is read, so a whole-genome mask holds no whole-genome array. The mask carries the `reference` coordinate of the reference that was used.

`pbzarr.write(target, variables, *, encoding=None, storage_options=None)` adds `xr.DataArray` values as new root arrays through `to_zarr(mode="a")`. `target` is a path, a URL, or a Dataset from `open`; for a Dataset, the store is `ds.encoding["source"]`. `variables` is a mapping of name to `xr.DataArray`, and an `xr.Dataset` is such a mapping. The write reads the store metadata when it is called and checks the values against it. It keeps no cache. It never changes an existing array, and it returns the store opened again. The Dataset passed as `target` does not change. A genomic dimension must come first and span the reference. Shared non-genomic dimensions must match the store's labels. A new dimension needs a label coordinate that lies on that dimension only. The write finds the `reference` coordinate by name, with or without its index, and compares it with the reference of the store. A different reference raises `AlignmentError`. Then the write drops the coordinate. A value without the coordinate is accepted when its genomic length matches. Tracks are numeric or bool. A variable with no genomic dimension can also hold strings, which are stored as variable-length UTF-8. The write saves the non-dimension coordinates of the values, such as `sample_group(sample)`. A coordinate with the name of a store array must have the same dimensions and labels as that array. The write drops every 0-d coordinate, for example `sample` after `isel(sample=0)` or `quantile` after `quantile(0.5)`. The write refuses a coordinate over `position` or `bin_N`. The write keeps the root attributes, drops `pbz:summary` and the zarrs provenance attribute from the values, refuses other `pbz:` attributes and attributes that change decoding, and writes `_FillValue` only when the caller declares one in `encoding=`. Track names must not be genomic dimension names or `reference`. It removes the arrays it created if the write fails. Written tracks use the Zarr and xarray default codecs unless `encoding=` says otherwise. The write checks the store by listing it, not by reading the consolidated copy. As its last step, it replaces the consolidated copy with `zarr.consolidate_metadata`.

`create(destination, *, reference, storage_options=None)` creates an empty store at a local path for `write` and returns it opened as an `xr.Dataset`. It does not create remote stores yet: a URL destination or a `storage_options` value other than `None` raises `ValueError` before anything is written. `import_d4`, `import_bigwig`, `import_bed`, and `import_bam` follow the store-creation rule of the CLI (see [CLI](#cli)). They write local stores only, and return a report dictionary with `positions`, `chunks_written`, `chunks_skipped`, `bytes_written`, and `wall_seconds`. They share the `dim`, `threads`, `chunk_size`, `column_chunk_size`, `shard_size`, `shard_column_size`, `codecs`, and `progress` keyword arguments. `import_bam` takes the CRAM reference FASTA as `fasta=`.

`import_bed(destination, sources, *, genome=None, ...)` covers every BED shape in one call, mirroring `pbz import bed`: `track=`/`column=`/`dtype=` import one BED column as one track, and `schema=` maps BED column selectors (header names or 1-based numbers) to dtypes, one track per entry, with `None` dtypes inferred. Any 2D output, over several sources or over several BED columns of one wide track, requires `dim=`. `genome=` is needed only to create the store.

## Rust API

`Dataset::create(path, reference)` and `Dataset::open(path)` operate on filesystem stores. `create_with_storage` and `open_with_storage` take any synchronous readable, writable, listable Zarr storage handle. `TrackConfig::new(dtype)` configures scalar or 2D tracks with generic columns, chunking, optional sharding, fill value, and optional description or source metadata. `Dataset::read_region<T>` reads one region of a track at full column width.

`Reference` holds named contigs and derives flat offsets. Regions are 0-based and half-open.

The crate root exports `Dataset`, `Reference`, `GenomicAxis`, `Contig`, `Region`, `TrackConfig`, `Import`, `Measurement`, and the error type `PbzError`. Three modules stay public: `pbzarr::import` holds the import configuration and targets, `pbzarr::io` holds the `ValueReader` trait for reader crates, and `pbzarr::stat` holds the region statistics.

An import refuses a track or dimension name that the store keeps for itself: `position`, any `bin_N`, `contig`, `contig_id`, `contig_length`, `contig_md5`, and `reference`.

The format-agnostic import engine drives any `ValueReader`. A reader exposes an ordered schema of named dtype fields. `read_into` fills one sink per field for one contig window in a single decode pass. `read_windows` decodes a run of adjacent windows and reports which windows have no data, so the engine can skip them. `fork` gives each worker its own handle.

`Import::from_readers(..)` starts a run: the builder chains `into_dataset(..)`, an optional axis statement (`fields_as_tracks()`, `fields_as_columns()`, or `readers_as_columns()`), and `options(..)`, and `run()` returns a `Report`. The public configuration surface is `pbzarr::import::{Config, Report, Source}`. Reader crates provide d4 (`int32`), bigWig (`float32`), BED (schema-typed columns), and BAM/CRAM (`int32` depth and composition counts) sources. Python exposes them through the destination-oriented writers.

## Interoperability

PBZ uses standard Zarr v3 groups, arrays, codecs, and variable-length UTF-8 strings. Released xarray opens a format 0.5 store directly through `xr.open_zarr`. The consolidated copy in the root `zarr.json` lets `consolidated=True` find every array, also for a reader that cannot list the store.

`pbzarr.open` adds identity and reference checks, Dask chunk defaults, region selection, region operations, and validated writes.

Cross-language validation (`pixi run validate-roundtrip`) imports d4 fixtures with Rust, opens the stores through released xarray and `pbzarr.open`, compares variables, dimensions, dtypes, root attributes, values, and a contig boundary, then writes a derived track from Python and checks the store with the Rust `check_dataset` example.

## CLI

The `pbz` CLI ships `import d4`, `import bigwig`, `import bed`, and `import bam`. The import subcommands take positional `PATH[:LABEL]` sources or `--file-list`, plus `--dim`, `--codecs`, `--decode-chunks`, and `-t`/`--threads`. `import bed` also takes a `--schema` TSV. `import bam` takes the CRAM FASTA as `--fasta`. When the output path does not exist, an import creates a format 0.5 store. The import first checks that all sources have the same contigs, then takes the store reference from them. If the import fails, it removes the new store. When the output path exists, the import adds tracks to that store, and the sources must match the store reference. The import refuses a format 0.4 store. The import refuses a track name that the store already uses. BED files have no contig lengths, so `import bed` needs `--genome` only to create a store; an existing store gives its own lengths. In Rust, `Target::Path` holds this rule, and `store_reference` gives the contig list for BED.

`view` exports one track as bedGraph-like TSV: `chrom`, `start`, `end`, then one value column per track column, with equal-value runs collapsed and rows in genome order. Positions whose values are all missing (NaN fill) are skipped. It subsets rank-2 columns with `--columns` and writes BGZF when the output path ends in `.gz`.

## Deferred work

Benchmarks, append from Python, and cloud write adapters remain future work.

Rectilinear chunk-grid opening needs upstream support and is not a current PBZ read mode. On-disk region reads and writes are likewise not part of the Python or PyO3 surface.
