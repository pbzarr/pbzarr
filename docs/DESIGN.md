# PBZ design

PBZ is a small Zarr v3 convention for dense, per-base genomic data. The format supplies genomic interpretation and a flat track layout; Zarr supplies storage and codecs; xarray and Dask supply the Python read and analysis model.

Format 0.5 (one root Dataset) is what imports write and what Python reads; see the [format 0.5 proposal](proposals/format-05/README.md). The 0.4 per-track layout described in parts of this document is legacy: the Rust `PbzStore` API, `pbz view`, `pbz stat`, and pyramids still use it. Earlier contig-major and wrapper-oriented designs are historical only.

## Development reader

The additive Rust `pbzarr::dataset::Dataset` reads the
[format `0.5` root-Dataset schema](proposals/format-05/README.md).
It catalogs direct root arrays, validates reference and dimension coordinates,
and reads typed rectangular subsets. `read_raw` preserves physical values and
missing sentinels; `read_bool` decodes the supported Boolean encodings.
The module caches metadata and reference geometry, so callers must reopen after
external changes. It does not provide a concurrent-mutation snapshot.

CLI imports and appends write format `0.5`, and `pbz view` and `pbz stat`
read both layouts. Python opens a `0.5` store through the `pbzarr.Dataset`
handle and refuses `0.4` stores. `PbzStore` remains the `0.4` Rust API, and
pyramids exist only there. A format `0.4` root presented to
`Dataset` reports as an unsupported format.

## Goals

PBZ makes per-base depth, signal, masks, and similar position-indexed data portable across Rust, Python, and ordinary Zarr tooling. It aims to preserve rank, keep genomic geometry explicit, and make xarray the normal Python read surface.

The format serves scalar signals as well as wide column tracks. A column axis can represent samples, strands, methylation contexts, or another declared label set; “cohort” is only the sample-specific case.

PBZ does not define a new container, codec, coordinate conversion layer, or query language. Coordinates are 0-based and half-open. Callers convert external 1-based formats at their own boundaries.

## Per-track data model

A collection is a set of named tracks. A track owns its own genome: ordered `(contig name, length)` pairs, a decorative optional genome name, and a checksum that identifies its geometry.

A scalar track has `values` shaped `(position,)`. A 2D track has `values` shaped `(position, column)`, where `column` is the declared dimension name and has a corresponding string coordinate array.

Track genomes are intentionally independent. A collection can contain tracks with different contigs or lengths; only tracks with the same genome checksum can be composed into one xarray Dataset.

The genome checksum is `md5:` followed by the MD5 of the canonical `"{name}\t{length}\n"` payload, with pairs sorted in byte-order by name. `genome_name` is decorative and does not participate in identity.

Every `values` array declares a fill value: 0 for integer tracks, NaN for float tracks, false for bool tracks. The d4, BED, and BAM/CRAM importers fill uncovered positions with 0. A bigWig stores no value at an uncovered base, so bigWig tracks import as float32 with a NaN fill. xarray reductions skip NaN by default, so uncovered bases do not bias region statistics.

## Flat v0.4 layout and metadata

The `.pbz` root is a bare Zarr group. Its attributes contain a `zarr_conventions` entry whose `name` is `"perbase"`; it does not carry a genome, track map, contigs, or offsets.

Each direct child is a track group. The child group has its own `zarr.json`, `values`, `offsets`, `contigs`, and, for 2D tracks, a column-coordinate array named after the second dimension.

```text
example.pbz/
├── zarr.json                  # root marker + consolidated metadata
├── depth/
│   ├── zarr.json              # perbase: interpretation attributes
│   ├── values                 # (ΣL, n_columns), (position, sample)
│   ├── offsets                # int64[k + 1], (contig_boundary,)
│   ├── contigs                # vlen-utf8[k], (contig,)
│   ├── sample                 # vlen-utf8[n_columns], (sample,)
│   └── scales/16/mean         # optional pyramid level, float32, (bin, sample)
└── mask/
    ├── zarr.json
    ├── values                 # (ΣL,), (position,)
    ├── offsets
    └── contigs
```

`offsets` is a prefix-sum ragged index. For `k` contigs, `offsets[0] == 0`, `offsets[k] == ΣL`, and contig `i` occupies flat positions `[offsets[i], offsets[i + 1])`. There is no `contig_lengths` array.

Track metadata contains the convention marker plus these interpretation fields:

```json
{
  "zarr_conventions": [{"uuid": "b7e3c1a2-...", "name": "perbase"}],
  "perbase:version": "0.4",
  "perbase:genome_checksum": "md5:...",
  "perbase:genome_name": "hg38",
  "perbase:ragged_index": "offsets",
  "perbase:ragged_contigs": "contigs",
  "perbase:coordinates": "0-based-half-open"
}
```

Optional `perbase:description` and `perbase:source` annotate a track. Dtype, fill value, chunk grid, shard grid, dimension names, and coordinate labels remain in ordinary Zarr array metadata rather than being duplicated here.

A writer creates arrays first and writes the track’s `perbase:` interpretation block last. The marker is therefore a completion record: discovery skips a child without it because it may be a partial import or unrelated Zarr group.

A track can carry an opt-in multiscale pyramid. Each level is a float32 array of per-bin means at `<track>/scales/<factor>/mean`. The writer bins each contig independently, so no bin crosses a contig boundary, and the last bin of a contig can be short. On NaN-fill tracks the mean skips NaN positions, and an all-NaN bin is NaN. The writer builds levels only after the base track publishes, and it stores the `multiscales` track attribute last, as the pyramid’s completion marker. Publication also seals the base: the library rejects later `values` writes on a track with a published pyramid.

The root `zarr.json` also carries a consolidated metadata map in the zarr-python v3 flavor: every node’s metadata document, inlined verbatim. Writers refresh the map on each track or pyramid publication. The copy is advisory: the per-node `zarr.json` documents stay authoritative.

## Chunks and shards

By default, every `values` array uses Blosc with zstd level 5 and byte shuffle. Scalar tracks are chunked by position; 2D tracks are chunked by position and column so compression can exploit similarity across columns. Import entry points accept an explicit Zarr v3 codec spec (`--codecs` on the CLI, `codecs=` in Python) that replaces this default: a standard `codecs` list, with sharding expressed as a `sharding_indexed` codec and an optional `chunk_grid` member for the outer grid.

Sharding is optional and has an outer position-first shard grid with full declared column width by default. The normal Zarr chunk shape is retained as the shard’s subchunk shape. Sharded writes append encoded subchunks through Zarr partial encoding instead of rewriting whole shards.

One import engine drives every format. A chunk is one inner position chunk, and it is the unit the engine writes. A span is a run of consecutive chunks, and it is the window one reader decodes in one pass. By default the engine sizes the span automatically: up to 4 million positions, and fewer when that leaves the workers short of pieces. A piece is one reader and one span. Each reader decodes each span once and fills every schema field in that one pass. Buffers still close and write per chunk, so the engine still skips an uncovered chunk inside a covered span. The engine pre-fills every buffer with the destination track’s declared fill value.

The worker that completes a buffer writes it. On an unsharded track, the buffer is one whole chunk. On a sharded track, the worker compresses the buffer and appends it to its shard exactly once, and a per-shard lock serializes the appends. The engine never writes a chunk that no reader covers. An absent chunk reads back as the fill value.

## Python API

The public namespace is `Dataset`, `Reference`, `PbzError`, `open`, `create`, `RegionQuery`, `parse_region`, `import_d4`, `import_bigwig`, `import_bed`, and `import_bam`. Python reads through zarr-python and xarray; Rust is used only for imports.

`pbzarr.open(source, *, mode="r", tracks=None, chunks=..., storage_options=None)` returns a `pbzarr.Dataset` handle for a format 0.5 store. It checks the pbz convention entry and version, and refuses a format 0.4 store. `mode="r+"` allows `write`. `tracks=` narrows the exposed tracks and keeps the reference and label coordinates. By default each track is a Dask array chunked by its stored shard (or chunk) shape; `chunks=None` gives lazy NumPy-backed arrays.

The handle is not a value. `ds[name]` returns a plain `xr.DataArray`, and `to_xarray()` returns the opened `xr.Dataset`. The handle owns the store, its `Reference`, and the operations that need store state: `regions`, `reduce_regions`, and `write`.

`Reference` is a mapping of contig name to length. `rows(query, dim="position")` maps one region to a slice of flat rows with the same rule as the Rust reader; on `bin_N` it gives the bins that intersect the region. Empty and out-of-range regions are refused. Select with `ds["depth"].isel(position=ds.reference.rows("chr1:100-200"))`. `sel(position=...)` is reserved: with no genomic index, xarray treats it as row positions, and its meaning changes when the index arrives.

`regions(intervals)` packs many non-overlapping regions into a derived Dataset with interval coordinates and `offsets`. `reduce_regions(intervals, reducer, **kwargs)` reduces each complete interval along `position`. Intervals may be a dataframe, three arrays, rows, or a list of region strings. Supported reducers are `mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `median`, `quantile`, and `summit` (which requires a `by=` variable). Only tracks with a leading `position` dimension take part, and they may have at most one more dimension.

An eager packed Dataset must still fit RAM: its complete values require memory proportional to selected bases × columns × variables. Default Dask is recommended for large selections.

`write(tracks, *, encoding=None)` adds `xr.DataArray` values as new root tracks through `to_zarr(mode="a")`. A genomic dimension must come first and span the reference. Shared non-genomic dimensions must match the store's labels, and a new dimension needs a labeled coordinate. The write keeps the root attributes, drops `pbz:summary` and the zarrs provenance attribute from the values, refuses other `pbz:` attributes and attributes that change decoding, and writes `_FillValue` only when the caller declares one in `encoding=`. Track names must not be genomic dimension names. It refuses when the store changed since the handle opened it, and it removes the arrays it created if the write fails. Written tracks use the Zarr and xarray default codecs unless `encoding=` says otherwise. The write leaves the store without consolidated metadata, as the Rust writer does.

`create(destination, *, reference)` creates an empty store and returns an `r+` handle. The reference must equal the genome of every source imported later: the same names, lengths, and order. `import_d4`, `import_bigwig`, `import_bed`, and `import_bam` write tracks into an existing format 0.5 store and return a small report dictionary. They share the `workers`, `chunk_size`, `column_chunk_size`, `shard_size`, `shard_column_size`, `codecs`, and `progress` keyword arguments. `import_bam` takes the CRAM reference FASTA as `fasta=`.

`import_bed(destination, sources, *, genome, ...)` covers every BED shape in one call, mirroring `pbz import bed`: `track=`/`column=`/`dtype=` import one BED column as one track, and `schema=` maps BED column selectors (header names or 1-based numbers) to dtypes, one track per entry, with `None` dtypes inferred. Any 2D output, over several sources or over several BED columns of one wide track, requires `column_dim=`.

## Rust API

`PbzStore::create(path)` and `PbzStore::open(path)` operate on filesystem stores, with `create_with_storage` and `open_with_storage` accepting an arbitrary synchronous readable, writable, listable Zarr storage handle.

`PbzStore::create_track(name, genome, config)` is the manual-authoring path. `TrackConfig::new(dtype)` configures scalar or 2D tracks with generic columns, chunking, optional sharding, fill value, and optional description or source metadata.

`Genome` owns named contigs and derives offsets and the checksum. `Track::read_region<T>` and `Track::write_region<T>` use 0-based, half-open regions and preserve the track rank at runtime-checked dtype `T`.

The format-agnostic import engine drives any `ValueReader`. A reader exposes an ordered schema of named dtype fields. `read_into` fills one sink per field for one contig window in a single decode pass. `may_have_data` lets the engine skip windows with no data, and `fork` gives each worker its own handle.

`Import::from_readers(..)` starts a run: the builder chains `into_track(s)(..)`, an optional axis statement (`fields_as_tracks()`, `fields_as_columns()`, or `readers_as_columns()`), and `options(..)`, and `run()` returns a `Report`. `estimate()` validates the same routing and predicts the physical layout and resource use without a run. The public configuration surface is `pbzarr::import::{Config, Report, Source}`. Reader crates provide d4 (`int32`), bigWig (`float32`), BED (schema-typed columns), and BAM/CRAM (`int32` depth and composition counts) sources. Python exposes them through the destination-oriented writers.

## Interoperability

PBZ uses standard Zarr v3 groups, arrays, codecs, and variable-length UTF-8 strings. Released xarray opens a format 0.5 store directly through `xr.open_zarr(..., consolidated=False)`.

`pbzarr.open` adds identity and reference checks, Dask chunk defaults, region selection, region operations, and validated writes.

Cross-language validation (`pixi run validate-roundtrip`) imports d4 fixtures with Rust, opens the stores through released xarray and `pbzarr.open`, compares variables, dimensions, dtypes, root attributes, values, and a contig boundary, then writes a derived track from Python and checks the store with the Rust `check_dataset` example.

## CLI

The `pbz` CLI ships `import d4`, `import bigwig`, `import bed`, `import bam`, and `scale` for existing tracks. The import subcommands take positional `PATH[:LABEL]` sources or `--file-list`, plus `--codecs`, `--scales`, and `--threads`. `import bed` also takes a `--schema` TSV. An import creates a format 0.5 store, or adds tracks to an existing store in whichever format that store already uses.

`view` exports one track as bedGraph-like TSV: `chrom`, `start`, `end`, then one value column per track column, with equal-value runs collapsed and rows in genome order. Positions whose values are all missing (NaN fill) are skipped. It subsets rank-2 columns with `--columns` and writes BGZF when the output path ends in `.gz`.

## Deferred work

Benchmarks, append from Python, cloud write adapters, and pyramid unpublish or rescale remain future work.

Rectilinear chunk-grid opening needs upstream support and is not a current PBZ read mode. On-disk region reads and writes are likewise not part of the Python or PyO3 surface.
