# PBZ: Per-Base Zarr

**Version 0.1**

**Date: 2026-05-28**

---

## Abstract

PBZ (Per-Base Zarr) is a convention for storing per-base resolution genomic signal data: read depths, methylation rates, accessibility masks, and similar continuous- or boolean-valued tracks indexed by genomic position. It is layered on top of Zarr v3 and inherits Zarr's chunking, compression, and concurrent-write semantics. PBZ exists because the per-sample-file model used by D4 and bigWig does not compress well across cohorts and forces cross-sample computation through Python loops. PBZ spans a spectrum rather than serving cohorts alone: a single-sample store is a first-class artifact competitive with D4, and many single-sample stores combine into one chunked array indexed by (position, sample) where cross-sample compression and vectorized math pay off. The result is exposed as a regular xarray Dataset. This document specifies the on-disk layout, the metadata schema, and the I/O conventions that any conforming implementation must satisfy. Two implementations are described, a Rust library built on the `zarrs` crate and a Python wheel built on `zarr-python`, and the format is shown to round-trip identical stores across the two.

---

## 1. Introduction

Per-base genomic signal data (read depths, methylation rates, accessibility masks, mappability scores) is large, structured, and increasingly central to downstream bioinformatics workflows. The existing storage landscape was built when "one signal per file" was a reasonable default. bigWig and D4 both encode a single sample's signal as an indexed binary file optimized for region queries on that one sample. Both formats predate the chunked-array tooling (Zarr, Xarray, Dask) that the broader scientific-computing ecosystem has converged on, and both treat multi-sample and multi-track workloads as a concern outside the file format.

PBZ takes a different starting point. Per-base signals are naturally chunked along the position axis, often carry an additional axis (sample, strand, methylation context, mask category), and benefit from compression that operates across array dimensions rather than file-at-a-time. These are properties Zarr v3 already provides. PBZ is the smallest convention layer needed to put per-base genomic data into Zarr v3 in a way that is interoperable across languages and ergonomic to read with xarray.

A similar approach exists in a different genomics domain. [cooler](https://github.com/open2c/cooler) defines a layout for sparse Hi-C contact matrices on HDF5: a fixed schema for bins, pixels, and indexes, with all I/O and compression delegated to the host library. PBZ does the same thing for per-base signal data, building on Zarr v3 rather than HDF5. The shape is dense (position by column) rather than sparse (bin by bin), but the convention-over-implementation stance is the same.

Three classes of workload motivate the format and inform its defaults:

* **Region queries across many tracks.** A typical viewer or downstream pipeline wants depth, mask, and annotation tracks for the same region together. Holding them in one store, with chunks aligned on the position axis, makes this a single open and a single slice rather than a sweep across `O(tracks)` separate files.
* **Cohort-shaped computation.** Operations like "mark sites where ≥K samples have depth ≥D" or per-position cross-sample summaries are awkward against per-sample files (d4-format#82, clam#25), and per-sample compression does not exploit the high redundancy across a cohort (d4-format#64). Stored as a (position, sample) chunked array, these operations become vectorized slab reads with compression operating on two-dimensional blocks. This is the workload where PBZ pays off most clearly.
* **xarray-native downstream code.** Once data is in a Zarr store with sensible dimension names, the read path is `xr.open_datatree(...)`. There is no bespoke client, no per-format SDK, and no Python loop separating the user from `numpy` / `dask` / `xarray` operators.

PBZ is best understood as a spectrum keyed on one axis: the column-chunk width of a track. A single-sample store has no column axis at all and is a complete, first-class artifact, competitive with D4 and bigWig on size and region-query latency while already living in the Zarr/xarray world. Combining many single-sample stores produces a wide-column-chunked cohort array, the regime where cross-sample compression and per-position cross-sample computation pay off. The cohort is the high-value end of this spectrum, not the whole point: the format earns its place for a single sample, then compounds as samples accumulate, and the downstream code stays in the xarray world throughout. Incremental growth of a cohort by appending samples is a near-term goal rather than a v0.1 capability; see §9.

### 1.1. Design Goals

* **Convention over implementation.** PBZ is a layout and metadata schema on top of Zarr v3. It does not define a new container format, codec, or storage abstraction; Zarr v3 already provides those.
* **Multi-axis tracks.** Tracks may be one-dimensional (one value per position) or two-dimensional (a vector per position, indexed by an arbitrary column dimension). Cohorts indexed by sample are the prototypical 2D case; strands, methylation contexts, and mask categories fit the same shape.
* **xarray as the read API.** Dimension names and coord arrays follow conventions the xarray-zarr backend already understands. Reads do not require a PBZ-specific client.
* **Cross-language parity.** A store written by any conforming implementation must be readable by every other, byte for byte.
* **0-based, half-open coordinates everywhere.** No 1-based coordinates anywhere in the on-disk representation. Conversion happens only at the I/O boundary against external formats.

### 1.2. Non-Goals

* **No on-the-fly resampling or multi-resolution.** PBZ does not maintain coarsened views of a track. Multi-resolution would require promoting tracks to groups; this is deferred and would be a breaking layout change if it ever lands.
* **No internal coordinate-system conversion.** The library does not translate between assemblies or 1-based formats; callers do that at import.
* **No bespoke compression.** Compression is delegated to Zarr v3 codecs.

---

## 2. Data Model

A PBZ store represents one or more **tracks** over a fixed **genome**. The genome is an ordered list of contigs and their lengths. A track is a function over genomic positions of a single dtype; cohort tracks add a second axis for columns (most commonly samples).

### 2.1. Genome

A genome is a tuple `(contigs, contig_lengths)` where `contigs` is an ordered list of contig names (e.g., `chr1`, `chr2`, ..., `chrM`) and `contig_lengths` is the corresponding list of contig lengths in base pairs. Contig order is significant: it defines the iteration order for whole-genome operations and the index of contig names in the `contigs` array. Names are stored as variable-length UTF-8.

### 2.2. Tracks

Two track shapes are defined:

* **1D tracks.** A single value per position. The position axis carries no explicit coordinate; a 0-based integer index into the contig is implied. Examples: a boolean accessibility mask, a per-position mappability score, a single sample's read depth.
* **2D (cohort) tracks.** A vector of values per position, with the vector index drawn from a writer-declared column axis. The writer names the dimension (`column_dim`) and supplies the column labels in metadata. "Cohort" is shorthand for the column-axis pattern in general: the prototypical case is samples, but strands, methylation contexts, and mask categories fit the same shape and are first-class.

The choice of 1D vs 2D is per track, not per store. A single store typically mixes both: a 2D depth track over samples next to a 1D accessibility mask, a 1D mappability score, and so on, all addressable against the same genome.

### 2.3. Dtypes

PBZ tracks support the following dtypes:

| Family   | Allowed types                |
|----------|------------------------------|
| Unsigned | `uint8`, `uint16`, `uint32`  |
| Signed   | `int8`, `int16`, `int32`     |
| Float    | `float32`, `float64`         |
| Boolean  | `bool`                       |

All values within a single track share one dtype. Mixed-dtype tracks must be split into separate tracks.

---

## 3. On-Disk Layout

A PBZ store is a Zarr v3 hierarchy with the extension `.pbz`. The root group carries store-level metadata; one subgroup exists per contig; tracks are arrays inside the contig groups.

```
foo.pbz/
├── zarr.json                  # root group metadata, including perbase_zarr attr block
├── contigs                    # 1D string array, dim=[contigs]
├── contig_lengths             # 1D int64 array, dim=[contigs]
├── chr1/
│   ├── zarr.json              # group metadata
│   ├── sample                 # 1D string coord, dim=[sample]; present iff a cohort track uses dim "sample"
│   ├── depth                  # 2D, shape=(chr1_length, n_samples), dims=[position, sample]
│   └── mask                   # 1D, shape=(chr1_length,),           dims=[position]
└── chr2/
    └── ...
```

### 3.1. Contig-Major

Tracks live inside contig groups, addressed as `<contig>/<track>`. A region query resolves to a single open and a single slice on one array per track. Whole-genome operations iterate the contig groups in the order given by the root `contigs` array.

### 3.2. Tracks Are Arrays

Each track is realized as one Zarr array per contig at the path `<contig>/<track>`. Track-level metadata lives in the root `perbase_zarr.tracks[name]` block; the per-contig arrays carry only the standard Zarr v3 array metadata (shape, chunk grid, codecs, fill value, dimension names).

### 3.3. Coord Arrays

A cohort track with `column_dim = "sample"` requires a 1D string array at `<contig>/sample` listing the column labels. The same `column_dim` must carry identical labels across every contig in the store; this invariant is established by the writer at track-creation time and is not currently re-validated by readers.

Scalar tracks have no coord array. Their only dimension is `position`, which is implicitly indexed.

### 3.4. Compression

Every data array is compressed with **Blosc** in zstd mode (level 5) with byte shuffle by default. The codec pipeline is recorded per array in standard Zarr v3 metadata (`zarr.json`), not in `perbase_zarr`, and is fixed when the array is created. A writer that fills an existing array, for example a cross-language importer populating tracks another tool created, encodes through the pipeline already recorded on the array rather than selecting its own. A consequence for interop: any codec used must be one that every implementation reading or writing the store can both encode and decode. The default Blosc(zstd-5, byte-shuffle) satisfies this on both `zarrs` and `zarr-python`.

### 3.5. Sharding

Sharding is off by default. It may be enabled per track via the `shard_size` and `shard_column_size` metadata keys. The default may flip in a later revision once the sharding sweep benchmark produces guidance.

---

## 4. Metadata Schema

PBZ metadata lives in the root group's `attributes` object under a single namespaced key, `perbase_zarr`. Per-track metadata is centralized at the root rather than scattered across per-track-group attrs, so a reader sees every track's configuration in one place.

### 4.1. Root Attribute

```json
{
  "perbase_zarr": {
    "version": "0.1",
    "coordinate_space": "GRCh38",
    "tracks": {
      "depth": {
        "dtype": "uint16",
        "chunk_size": 1000000,
        "column_dim": "sample",
        "column_chunk_size": 16
      },
      "mask": {
        "dtype": "bool",
        "chunk_size": 1000000
      }
    }
  }
}
```

| Key                | Required | Notes                                                                                       |
|--------------------|----------|---------------------------------------------------------------------------------------------|
| `version`          | yes      | Spec version. `0.1` at the time of writing.                                                 |
| `coordinate_space` | optional | Free-form string identifying the reference assembly (e.g., `GRCh38`, `hg19`, `T2T-CHM13`).  |
| `tracks`           | yes      | Object mapping track name to per-track metadata. See §4.2.                                  |

### 4.2. Per-Track Metadata

| Key                 | Required             | Default            | Notes                                                            |
|---------------------|----------------------|--------------------|------------------------------------------------------------------|
| `dtype`             | yes                  | n/a                | One of the allowed dtypes (§2.3).                                |
| `chunk_size`        | yes                  | 1,000,000          | Chunk extent along the position axis.                            |
| `column_dim`        | iff cohort           | absent ⇒ 1D scalar | Dimension name (e.g., `"sample"`, `"strand"`).                   |
| `column_chunk_size` | iff cohort           | 16                 | Chunk extent along the column axis.                              |
| `shard_size`        | optional             | absent (off)       | Position-axis shard size, in chunks.                             |
| `shard_column_size` | iff sharded & cohort | n/a                | Column-axis shard size, in chunks.                               |
| `fill_value`        | optional             | dtype-natural      | `0` for ints, `NaN` for floats, `false` for bool.                |
| `description`       | optional             | none               | Human-readable description.                                      |
| `source`            | optional             | none               | Tool plus version that wrote the track.                          |
| (other keys)        | preserved            | n/a                | Round-tripped verbatim. Tool-specific keys SHOULD be namespaced. |

Implementations MUST preserve unknown keys across read-modify-write cycles. Unknown top-level keys under `perbase_zarr` are similarly preserved.

---

## 5. I/O Conventions

### 5.1. Coordinates

All positional coordinates in PBZ, including region queries, write extents, and chunk boundaries, are **0-based and half-open**. A query `chr1:1000-2000` refers to positions `[1000, 2000)`, a 1000-base region. Conversion from 1-based formats (BED in some implementations, VCF when interpreting `POS`, etc.) is the caller's responsibility.

### 5.2. Region Queries

The canonical region-query form is `<contig>:<start>-<end>`. A region selects a contiguous slice on the position axis of a single contig; no multi-contig region query is defined at v0.1. Cohort tracks may be further sliced on the column axis by name (e.g., a single sample, a subset of samples).

### 5.3. Chunking

Default chunk shape is `(chunk_size, column_chunk_size)` for cohort tracks and `(chunk_size,)` for scalar tracks. The defaults of 1,000,000 positions × 16 columns are tuned for the human genome and typical cohort sizes; they are not load-bearing for correctness and may be overridden per track. Region reads project onto an integer number of chunks plus partial-chunk slices at the edges.

### 5.4. Concurrent Writes

Concurrent writes to non-overlapping chunks of the same Zarr array are safe under both `zarrs` and `zarr-python`. PBZ implementations partition import tasks one per chunk and rely on this guarantee. Concurrent writes to overlapping chunks are undefined behavior.

---

## 6. Library Surfaces

The format is implementation-agnostic. The two libraries described below are the reference implementations.

### 6.1. Rust (`pbzarr` crate)

The Rust crate is a single library crate organized in layered modules.

| Module                  | Responsibility                                                                |
|-------------------------|-------------------------------------------------------------------------------|
| `genome.rs`             | `Genome`, `Contig`, `ContigId`, `Region`. Pure types; no I/O.                 |
| `region_query.rs`       | `RegionQuery` and the `<contig>:<start>-<end>` parser.                        |
| `io/dtype.rs`           | `Dtype` tag, `Numeric` trait, dtype-to-Rust-type mapping.                     |
| `io/reader.rs`          | `ValueReader` trait. Import sources implement this.                           |
| `pbzarr-readers` crate  | `D4Reader` (int32) and `BigWigReader` (float32), the built-in `ValueReader`s, in a separate crate so the core stays free of git dependencies. |
| `store.rs`              | `PbzStore`. Owns the root group handle and the cached genome.                 |
| `track.rs`              | `Track`, `TrackConfig`, `TrackMetadata`. Caches per-contig array handles.     |
| `import/pipeline.rs`    | `run_pipeline`. crossbeam-channel work distribution across workers.           |
| `pbzarr-readers` crate  | `from_d4` / `from_bigwig`. Format-specific entry points on top of `run_pipeline`. |
| `error.rs`              | `PbzError`, derived via `thiserror`.                                          |

`Track` is not generic over dtype. The on-disk dtype is determined by metadata; runtime checks at `read_region<T>` and `write_region<T>` reject mismatches via `PbzError::DtypeMismatch`. The trade-off is one runtime check per region I/O against simpler types for callers that mix tracks of different dtypes in the same code path.

```rust
let mut store = PbzStore::create("out.pbz", genome, Some("GRCh38".into()))?;

store.create_track("mask", TrackConfig::new(Dtype::Bool))?;
store.create_track(
    "depth",
    TrackConfig::new(Dtype::U32)
        .columns(vec!["A".into(), "B".into(), "C".into()])
        .column_dim("sample"),
)?;

let region = Region { contig: store.genome().id("chr1").unwrap(),
                      start: 1_000, end: 2_000 };
let data: ArrayD<u32> = store.track("depth").unwrap().read_region(&region)?;
```

**Import pipeline.** `pbzarr::import::run_pipeline<T, R: ValueReader>` parallelizes import at the chunk level. It forks one reader per worker, partitions write tasks one-per-chunk across a bounded `crossbeam-channel`, and collects the first error into a shared `Mutex<Option<PbzError>>`. `zarrs` is safe for concurrent writes to non-overlapping chunks; the chunk-level partitioning preserves that invariant. Caller code sets `Config::workers` to choose parallelism; the library does not spawn its own pool.

**Concurrency model.** Synchronous against `zarrs::FilesystemStore`, with parallelism opt-in through the import pipeline. No `rayon`, no async runtime, no internal thread pools. Library code contains no `unwrap`, `expect`, or `panic!`; all failure modes surface as `PbzError`.

**Escape hatches.** `Track::zarr_array(contig)` borrows the underlying `zarrs::Array` handle for direct chunk-level access not exposed by `read_region` / `write_region`. Callers who need codec inspection, sharded subset reads, or custom chunk iteration drop down to `zarrs` without losing the metadata layer above.

### 6.2. Python (`pbzarr`)

The Python wheel is a maturin mixed project. The user-facing surface is the `PbzStore` class:

```python
import pbzarr

# One-shot from a source file: contigs and lengths come from the header, and
# the dtype is set by the format (d4 -> int32, bigWig -> float32).
store = pbzarr.PbzStore.from_d4("out.pbz",
                                {"A": "/data/A.d4", "B": "/data/B.d4"},
                                track="depth", column_dim="sample")

# Or build it explicitly.
store = pbzarr.PbzStore.create("out.pbz", contigs=["chr1"],
                               contig_lengths=[248_956_422],
                               coordinate_space="GRCh38")
store.create_track("depth", dtype="int32",
                   columns=["A", "B", "C"], column_dim="sample")
store.import_d4("depth", sources=[("/data/A.d4", "A"), ("/data/B.d4", "B")])
```

`PbzStore.create` and `create_track` are pure Python over `zarr-python`. `import_d4` / `import_bigwig` are thin PyO3 bindings around the Rust import pipeline that release the GIL during the operation, and `from_d4` / `from_bigwig` wrap create plus import in one call, reading the contig list from the source header through the native `d4_contigs` / `bigwig_contigs` helpers. Custom numpy writes use `zarr-python` directly (e.g., `g["chr1/depth"][:1000, :] = arr`) rather than a PBZ-specific wrapper, since the arrays are normal Zarr v3 arrays once registered.

### 6.3. Read API

The Python read API is an xarray accessor registered on `xarray.DataTree` when `pbzarr` is imported:

```python
dt = pbzarr.open("out.pbz")          # thin wrapper around xr.open_datatree
dt.pbz.tracks                         # ['depth', 'mask']
dt.pbz.region("chr1:1000-2000")       # xr.Dataset, one contig sliced on position
dt.pbz.region("chr1:1000-2000", track="depth", column="A")  # xr.DataArray
```

No PBZ-specific lazy machinery is required; dask integration falls out of `xr.open_datatree(..., chunks=...)`.

### 6.4. Command-Line Interface (Planned)

A command-line wrapper around the Rust library is planned for v0.2. The CLI covers operations that benefit from being invocable outside a programming environment: import, ad-hoc inspection, and export to text formats consumed by existing UNIX-shaped pipelines (`sort`, `awk`, `bgzip`, `tabix`, etc.).

Anticipated subcommands:

| Subcommand                                                  | Purpose                                                                                       |
|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `pbz import <input> <store> --track <name>`                 | Bulk import from d4 and bigWig today; bedGraph, BED, BAM/CRAM as readers are added.           |
| `pbz export <store> <region> --format tsv\|bed\|bedgraph`   | Stream a region to a text format. Stdout by default; `--output` for a file.                   |
| `pbz info <store>`                                          | Print store metadata: contigs and lengths, track names, dtypes, chunk shapes, sharding state. |
| `pbz region <store> <region>`                               | Quick region query; pretty-print the slab to stdout.                                          |
| `pbz validate <store>`                                      | Audit spec conformance: coord-array consistency, dtype tags, metadata schema.                 |

Each subcommand maps to a small wrapper around an existing library entry point; the CLI does not own format or schema decisions. v0.1 ships library-only because every documented workflow is expressible through the library APIs, and committing to a CLI command surface before the library API itself has stabilized would force premature interface decisions. The CLI returns once the library has absorbed a few rounds of downstream feedback.

---

## 7. Cross-Language Interop

The on-disk layout, metadata schema, dtype tags, dimension names, coord arrays, and codec configuration are all writer-agnostic. Concretely:

* **Variable-length UTF-8 string arrays** (Zarr v3 `vlen-utf8` codec) round-trip cleanly between `zarrs` (Rust) and `zarr-python` (Python). The `contigs` array and per-contig column-label arrays use this encoding.
* **Zarr v3 `dimension_names`** are written into array metadata directly, not as a separate `_ARRAY_DIMENSIONS` attribute. The xarray-zarr backend reads these without additional configuration.
* **Codec pipelines** live in each data array's own `zarr.json`, not in `perbase_zarr`, and are fixed when the array is created. A later writer that fills an existing array, including a cross-language importer populating tracks another tool created, encodes through the pipeline already recorded on the array rather than choosing its own. The interop constraint that follows: a store meant to be read or written across languages must use only codecs every implementation can both encode and decode. The default Blosc(zstd-5, byte-shuffle) satisfies this on both `zarrs` and `zarr-python`.
* **The `perbase_zarr` root attribute** is preserved verbatim by both implementations; unknown keys survive a read-write cycle.

Cross-language round-trip is enforced by a CI integration test that writes a fixture store from one language and reads it from the other.

---

## 8. Design Considerations

The Python wheel implements `PbzStore.create` and `create_track` in pure Python rather than reusing the Rust implementation via PyO3. The reason is that the small writes those functions perform offer no meaningful Rust speedup, and binding them through PyO3 forces every store-creation call to cross the FFI boundary and re-enter the Rust runtime. The cost of the pure-Python path is that on-disk layout knowledge lives in two places, mitigated by the cross-language round-trip test.

---

## 9. Open Questions and Future Work

* **Validation helpers.** Coord-array consistency across contigs is currently a writer guarantee. A `pbzarr::validate` helper that audits a store for spec conformance is on the list; readers currently trust the writer.
* **Append modes.** Appending contigs or columns to an existing store is not supported in v0.1. The data model does not preclude it, and it is the major focus moving forward. The planned approach for v0.2 represents staged columns as a cheap size-1 tail on the column axis (a rectilinear chunk grid), folded into the wide compacted chunks by a later compaction step, so appends never rewrite existing chunks. It is blocked on rectilinear chunk-grid support reaching a released xarray. zarr-python supports it in a released version today; the zarrs writer lands in an unreleased version, so the Rust side waits on that release.
* **Remote stores.** Synchronous I/O against local files is the v0 target. Remote backends are a big win with Zarr, and will be supported eventually.
* **Additional import formats.** d4 and bigWig are the built-in import formats today. bedGraph, BED, and BAM/CRAM are the obvious additions; each requires a `ValueReader` implementation and a corresponding `from_*` entry point in the `pbzarr-readers` crate.

---

## 10. Conclusion

PBZ is a thin convention on top of Zarr v3 that aligns per-base genomic data along the axis on which it is most often computed: position × sample, with samples co-located in compressed chunks. It is deliberately small in surface area, comprising a layout, a metadata schema, and a set of I/O conventions. The reference implementations described here demonstrate that the convention is round-trippable across languages and tractable to build against existing Zarr toolchains. The format's value will be determined empirically by the cohort-compression, mask-generation, and cross-sample-math benchmarks in the v0.1 suite; this document captures the design those benchmarks will validate.
