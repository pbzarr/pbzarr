# PBZ design

PBZ is a small Zarr v3 convention for dense, per-base genomic data. The format supplies genomic interpretation and a flat track layout; Zarr supplies storage and codecs; xarray and Dask supply the Python read and analysis model.

This document describes PBZ v0.4, the current layout and public APIs. Earlier contig-major and wrapper-oriented designs are historical only.

## Goals

PBZ makes per-base depth, signal, masks, and similar position-indexed data portable across Rust, Python, and ordinary Zarr tooling. It aims to preserve rank, keep genomic geometry explicit, and make xarray the normal Python read surface.

The format serves scalar signals as well as wide column tracks. A column axis can represent samples, strands, methylation contexts, or another declared label set; “cohort” is only the sample-specific case.

PBZ does not define a new container, codec, coordinate conversion layer, or query language. Coordinates are 0-based and half-open. Callers convert external 1-based formats at their own boundaries.

## Per-track data model

A collection is a set of named tracks. A track owns its own genome: ordered `(contig name, length)` pairs, a decorative optional genome name, and a checksum that identifies its geometry.

A scalar track has `values` shaped `(position,)`. A 2D track has `values` shaped `(position, column)`, where `column` is the declared dimension name and has a corresponding string coordinate array.

Track genomes are intentionally independent. A collection can contain tracks with different contigs or lengths; only tracks with the same genome checksum can be composed into one xarray Dataset.

The genome checksum is `md5:` followed by the MD5 of the canonical `"{name}\t{length}\n"` payload, with pairs sorted in byte-order by name. `genome_name` is decorative and does not participate in identity.

## Flat v0.4 layout and metadata

The `.pbz` root is a bare Zarr group. Its attributes contain a `zarr_conventions` entry whose `name` is `"perbase"`; it does not carry a genome, track map, contigs, or offsets.

Each direct child is a track group. The child group has its own `zarr.json`, `values`, `offsets`, `contigs`, and, for 2D tracks, a column-coordinate array named after the second dimension.

```text
example.pbz/
├── zarr.json                  # root zarr_conventions marker
├── depth/
│   ├── zarr.json              # perbase: interpretation attributes
│   ├── values                 # (ΣL, n_columns), (position, sample)
│   ├── offsets                # int64[k + 1], (contig_boundary,)
│   ├── contigs                # vlen-utf8[k], (contig,)
│   └── sample                 # vlen-utf8[n_columns], (sample,)
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

## Chunks and shards

Every `values` array uses Blosc with zstd level 5 and byte shuffle. Scalar tracks are chunked by position; 2D tracks are chunked by position and column so compression can exploit similarity across columns.

Sharding is optional and has an outer position-first shard grid with full declared column width by default. The normal Zarr chunk shape is retained as the shard’s subchunk shape; write work must align with the physical chunk or shard to avoid unsafe read-modify-write races.

The importer partitions the flat position axis into one full-column-width physical write unit per task: a chunk for regular arrays, or the shard at the Array level when sharded. Tasks may cross contig boundaries; each worker fills the overlaps by contig name and writes its exclusive whole unit, so no concurrent read-modify-write or write lock is needed.

## Python API

Importing `pbzarr` registers `.pbz` accessors on `xr.Dataset` and `xr.DataTree`. The public namespace is `PbzError`, `open`, `RegionQuery`, `parse_region`, `create_store`, `import_d4`, `import_bigwig`, `import_bed`, `import_bed_multi`, `import_bam`, and `stack` (deprecated).

`pbzarr.open(source, *, tracks=None, chunks=..., storage_options=None)` opens a track as an `xr.Dataset` or a collection as an `xr.DataTree`. `tracks=` selects direct collection children without opening others.

`DataTree.pbz.dataset(tracks=None)` composes selected compatible tracks without aligning labels. It rejects tracks with different genome checksums or incompatible shared coordinate labels.

`Dataset.pbz.region(query, *, column=None)` returns one contig-local `values` slice. `Dataset.pbz.regions(intervals)` and `DataTree.pbz.regions(intervals, *, tracks=None)` create a packed, derived Dataset for many non-overlapping regions.

Packed data carries interval coordinates and `offsets`. `Dataset.pbz.reduce_regions(intervals, reducer, **kwargs)` packs the selected intervals and reduces each complete interval along `position` in one call. Supported reducers are `mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `median`, `quantile`, and `summit` (which requires a `by=` variable).

Default opening turns regular Zarr chunks, or outer shards, into Dask chunks. `chunks=None` uses xarray’s non-Dask backend: ordinary one-region slices stay lazy, while many-region reads gather only selected pieces eagerly.

An eager packed Dataset must still fit RAM: its complete values require memory proportional to selected bases × columns × variables. Default Dask is recommended for large selections; xarray owns the resulting resource lifetime, so callers close returned datasets and trees when done.

Packed region Datasets are in-memory/lazy derived values. Rectilinear opening and an on-disk region read/write representation are explicitly deferred.

All writers are destination-oriented. `import_bam` returns a small report dictionary; the other writers return `None`. `create_store(destination)` creates an empty collection before `import_d4`, `import_bigwig`, `import_bed`, `import_bed_multi`, or `import_bam` writes tracks into it.

`stack(sources, destination, *, tracks=None, column_dim=None, column_chunk_size=None, workers=None)` creates a new destination collection from scalar tracks in source collections. Callers reopen any written destination with `pbzarr.open`. `stack` is deprecated (it emits `DeprecationWarning`); import all samples in one cohort import instead.

## Rust API

`PbzStore::create(path)` and `PbzStore::open(path)` operate on filesystem stores, with `create_with_storage` and `open_with_storage` accepting an arbitrary synchronous readable, writable, listable Zarr storage handle.

`PbzStore::create_track(name, genome, config)` is the manual-authoring path. `TrackConfig::new(dtype)` configures scalar or 2D tracks with generic columns, chunking, optional sharding, fill value, and optional description or source metadata.

`Genome` owns named contigs and derives offsets and the checksum. `Track::read_region<T>` and `Track::write_region<T>` use 0-based, half-open regions and preserve the track rank at runtime-checked dtype `T`.

The format-agnostic Rust import pipeline takes a `ValueReader` and writes a `Track`. Reader crates provide d4 (`int32`), bigWig (`float32`), BED (schema-typed columns), and BAM/CRAM (`int32` depth and composition) sources; Python exposes them through the destination-oriented writers.

## Interoperability

PBZ uses standard Zarr v3 groups, arrays, codecs, and variable-length UTF-8 strings. Released xarray can open the hierarchy directly through `xr.open_datatree(..., engine="zarr", consolidated=False)`.

`pbzarr.open` adds validation, track discovery, coordinate classification, Dask chunk defaults, composition, and genomic region operations. It does not require an alternative Python store object or a separate data model.

Cross-language validation writes a regular fixture with Rust, opens it through released xarray and `pbzarr.open`, and compares direct children, dimensions, coordinate arrays, attributes, representative values, and contig-boundary behavior.

## Deferred work

Multi-resolution tracks, benchmarks, append operations, cloud write adapters, and additional import formats remain future work. The `pbz` CLI ships `import bed` and `import bam`; its `view` subcommand is not yet implemented.

Rectilinear chunk-grid opening needs upstream support and is not a current PBZ read mode. On-disk region reads and writes are likewise not part of the Python or PyO3 surface.

The format currently has its binding design decisions in the repository ADRs. A formal standalone specification may later supersede this design document.
