# PBZ: Per-Base Zarr

**Format version 0.4**

**Date: 2026-07-24**

---

## Abstract

PBZ (Per-Base Zarr) is a convention for storing per-base resolution genomic signal data: read depths, methylation rates, accessibility masks, and similar continuous- or boolean-valued tracks indexed by genomic position. It is layered on top of Zarr v3 and inherits Zarr's chunking, compression, and concurrent-write semantics. PBZ exists because the per-sample-file model used by D4 and bigWig does not compress well across cohorts and forces cross-sample computation through Python loops. PBZ spans a spectrum rather than serving cohorts alone: a single-sample store is a first-class artifact competitive with D4, and many single-sample stores combine into one chunked array indexed by (position, sample) where cross-sample compression and vectorized math pay off. The result is exposed as a regular xarray dataset.

The layout is *flat and self-describing*: each track is a standalone Zarr group holding one contiguous `values` array over the concatenated genome, plus a ragged index (`offsets`, `contigs`) that maps genomic positions onto that array. A track carries its own genome and its own conformance metadata; the store is a bare container of tracks. This document specifies the on-disk layout, the metadata schema, and the I/O conventions any conforming implementation must satisfy. Two implementations are described, a Rust library built on the `zarrs` crate and a Python wheel built on `zarr-python`, and the format round-trips identical stores across the two.

---

## 1. Introduction

Per-base genomic signal data (read depths, methylation rates, accessibility masks, mappability scores) is large, structured, and increasingly central to downstream bioinformatics workflows. The existing storage landscape was built when "one signal per file" was a reasonable default. bigWig and D4 both encode a single sample's signal as an indexed binary file optimized for region queries on that one sample. Both formats predate the chunked-array tooling (Zarr, Xarray, Dask) that the broader scientific-computing ecosystem has converged on, and both treat multi-sample and multi-track workloads as a concern outside the file format.

PBZ takes a different starting point. Per-base signals are naturally chunked along the position axis, often carry an additional axis (sample, strand, methylation context, mask category), and benefit from compression that operates across array dimensions rather than file-at-a-time. These are properties Zarr v3 already provides. PBZ is the smallest convention layer needed to put per-base genomic data into Zarr v3 in a way that is interoperable across languages and ergonomic to read with xarray.

A similar approach exists in a different genomics domain. [cooler](https://github.com/open2c/cooler) defines a layout for sparse Hi-C contact matrices on HDF5: a fixed schema for bins, pixels, and indexes, with all I/O and compression delegated to the host library. PBZ does the same thing for per-base signal data, building on Zarr v3 rather than HDF5. The shape is dense (position by column) rather than sparse (bin by bin), but the convention-over-implementation stance is the same.

Three classes of workload motivate the format and inform its defaults:

* **Region queries across many tracks.** A typical viewer or downstream pipeline wants depth, mask, and annotation tracks for the same region together. Holding them in one store, with chunks aligned on the position axis, makes this a single open and a single slice rather than a sweep across `O(tracks)` separate files.
* **Cohort-shaped computation.** Operations like "mark sites where ≥K samples have depth ≥D" or per-position cross-sample summaries are awkward against per-sample files (d4-format#82, clam#25), and per-sample compression does not exploit the high redundancy across a cohort (d4-format#64). Stored as a (position, sample) chunked array, these operations become vectorized slab reads with compression operating on two-dimensional blocks. This is the workload where PBZ pays off most clearly.
* **xarray-native downstream code.** Once data is in a Zarr store with sensible dimension names, the read path is `xr.open_datatree(...)`. There is no bespoke client, no per-format SDK, and no Python loop separating the user from `numpy` / `dask` / `xarray` operators.

PBZ is best understood as a spectrum keyed on one axis: the column-chunk width of a track. A single-sample store has no column axis at all and is a complete, first-class artifact, competitive with D4 and bigWig on size and region-query latency while already living in the Zarr/xarray world. Combining many single-sample stores produces a wide-column-chunked cohort array, the regime where cross-sample compression and per-position cross-sample computation pay off. The cohort is the high-value end of this spectrum, not the whole point: the format earns its place for a single sample, then compounds as samples accumulate, and the downstream code stays in the xarray world throughout. Incremental growth of a cohort by appending samples is a near-term goal rather than a current capability; see §9.

### 1.1. Design Goals

* **Convention over implementation.** PBZ is a layout and metadata schema on top of Zarr v3. It does not define a new container format, codec, or storage abstraction; Zarr v3 already provides those.
* **Self-describing tracks.** A track carries everything needed to interpret it: its genome (contigs and lengths), its ragged index, its dtype and dimension names, and its conformance metadata. A track is portable on its own, independent of the store that holds it.
* **Multi-axis tracks.** Tracks may be one-dimensional (one value per position) or two-dimensional (a vector per position, indexed by an arbitrary column dimension). Cohorts indexed by sample are the prototypical 2D case; strands, methylation contexts, and mask categories fit the same shape.
* **xarray as the read API.** Dimension names and coord arrays follow conventions the xarray-zarr backend already understands. Reads do not require a PBZ-specific client.
* **Cross-language parity.** A store written by any conforming implementation must be readable by every other, and identity metadata (the genome checksum) must match byte for byte.
* **0-based, half-open coordinates everywhere.** No 1-based coordinates anywhere in the on-disk representation. Conversion happens only at the I/O boundary against external formats.

### 1.2. Non-Goals

* **No on-the-fly resampling or multi-resolution.** PBZ does not maintain coarsened views of a track. Multi-resolution would nest further under the track group; it is deferred and would extend the layout, not break it.
* **No internal coordinate-system conversion.** The library does not translate between assemblies or 1-based formats; callers do that at import.
* **No bespoke compression.** Compression is delegated to Zarr v3 codecs.
* **No store-level genome.** The store holds no shared genome and no union of contigs across tracks. Genome identity lives on each track.

---

## 2. Data Model

A PBZ store is a container of **tracks**. Each track is a self-describing unit over its own **genome**: an ordered set of contigs and their lengths. A track is a function over genomic positions of a single dtype; 2D tracks add a second axis for columns (most commonly samples). Two tracks in one store may cover different genomes.

### 2.1. Genome

A genome is an ordered list of `(contig name, length)` pairs. Contig order is significant: it defines the iteration order for whole-genome operations and the order of names in the track's `contigs` array. Names are stored as variable-length UTF-8. A genome belongs to exactly one track; there is no store-level genome object.

Two derived values follow from a genome:

* **`offsets`** is the prefix-sum flat-start index over the contigs: an `int64` array of `k+1` entries where `offsets[i]` is contig `i`'s start on the flat position axis, `offsets[0] == 0`, and `offsets[k] == ΣL` (the sum of all contig lengths). A contig's length is recovered as `offsets[i+1] - offsets[i]`; there is no separate lengths array.
* **`genome_checksum`** is an `md5` over the canonical `"{name}\t{length}\n"` join of the genome's `(name, length)` pairs, sorted by name in byte order, rendered as `"md5:" + hex`. It is the sole identity used to decide whether two tracks are mergeable (for example, whether N single-sample tracks can be stacked into one cohort). A separate, decorative `genome_name` (for example `hg38`) may label a genome for humans; it is excluded from the checksum and never consulted for compatibility.

### 2.2. Tracks

Two track shapes are defined:

* **1D (scalar) tracks.** A single value per position. The position axis carries no explicit coordinate; a 0-based flat index into the concatenated genome is implied, and the `offsets`/`contigs` index recovers the contig and within-contig position. Examples: a boolean accessibility mask, a per-position mappability score, a single sample's read depth.
* **2D (column) tracks.** A vector of values per position, with the vector index drawn from a writer-declared column axis. The writer names the dimension (`column_dim`) and supplies the column labels in a coord array. The prototypical case is a cohort of samples, but strands, methylation contexts, and mask categories fit the same shape and are first-class.

The choice of 1D vs 2D is per track. A single store typically mixes both: a 2D depth track over samples next to a 1D accessibility mask and a 1D mappability score.

### 2.3. Coverage and Fill

A track's **contig set** (which contigs it includes) is distinct from its **coverage** (which positions within those contigs actually carry data). A position with no data takes the track's **fill value**, set once at track creation. The meaning of fill is format-dependent and drives reduction semantics (§5.5): a `NaN` fill marks genuine missing data, while a numeric fill (for example `0` on a d4 depth track) is real data indistinguishable from a true zero.

### 2.4. Dtypes

PBZ tracks support the following dtypes:

| Family   | Allowed types                |
|----------|------------------------------|
| Unsigned | `uint8`, `uint16`, `uint32`  |
| Signed   | `int8`, `int16`, `int32`     |
| Float    | `float32`, `float64`         |
| Boolean  | `bool`                       |

All values within a single track share one dtype. Mixed-dtype tracks must be split into separate tracks. The built-in importers narrow this further by source format: d4 imports to `int32`, bigWig to `float32`, and BED to `int32` / `float32` / `bool`.

---

## 3. On-Disk Layout

A PBZ store is a Zarr v3 hierarchy with the extension `.pbz`. The store root is a bare marker group. Each track is a subgroup, sized to `ΣL` (the sum of its own contig lengths), with no per-contig subdivision.

```
foo.pbz/
├── zarr.json                   # root: zarr_conventions marker only, no genome
├── depth/
│   ├── zarr.json               # track group: perbase: conformance block
│   ├── values                  # 2D (ΣL, n_columns), dims=[position, sample] (2D track)
│   ├── offsets                 # int64[k+1], prefix-sum ragged index over contigs
│   ├── contigs                 # vlen-utf8[k], contig names in genome order
│   └── sample                  # vlen-utf8[n_columns], column labels (name = column_dim)
└── mask/
    ├── zarr.json
    ├── values                  # 1D (ΣL,), dims=[position] (scalar)
    ├── offsets
    └── contigs
```

### 3.1. Flat, Self-Describing Tracks

A track is a Zarr group, not a bare array. It holds a `values` array plus the arrays that describe it: `offsets`, `contigs`, and (for 2D tracks) one column-label array. All layout knowledge needed to interpret the track lives inside the group. Because each track owns its genome, the store has no `contigs` array, no `contig_lengths` array, and no cross-track union.

### 3.2. The `values` Array

`values` is rank-faithful: shape `(ΣL,)` with `dims = [position]` for a scalar track, and `(ΣL, n_columns)` with `dims = [position, <column_dim>]` for a 2D track. Axis 0, the position axis, is the concatenation of every contig in genome order; it is the only ragged axis, partitioned by `offsets`. A region on a contig maps to a flat span: for contig `i`, the region `[s, e)` is the flat slice `[offsets[i] + s, offsets[i] + e)`.

A 2D track's `values` is chunked on the column axis as well as the position axis, so cross-sample compression operates on two-dimensional blocks. Structural facts (dtype, chunk shape, dimension names, fill value, codecs) live on the `values` array's own Zarr metadata and are not duplicated into the conformance block.

### 3.3. The Ragged Index

`offsets` (`int64[k+1]`) and `contigs` (`vlen-utf8[k]`) together form the ragged index: `contigs[i]` is the name of the `i`-th contig, and `offsets[i] .. offsets[i+1]` is its half-open flat range on the position axis. `offsets[0]` is always `0` and `offsets[k]` is always `ΣL`. This pair replaces the separate contig and contig-length arrays of earlier layouts.

### 3.4. Coord Arrays

A 2D track writes a 1D `vlen-utf8` array named after its `column_dim` (for example `sample`), listing the column labels in column order. xarray promotes this array to a coordinate on the column axis. Scalar tracks have no coord array; their only dimension is `position`, which is implicitly indexed.

### 3.5. Compression

Every `values` array is compressed with **Blosc** in zstd mode (level 5) with byte shuffle. This is fixed, not configurable per store. The codec pipeline is recorded per array in standard Zarr v3 metadata (`zarr.json`), not in the conformance block, and is fixed when the array is created. A writer that fills an existing array encodes through the pipeline already recorded on that array rather than choosing its own. The interop constraint that follows: any codec used must be one that every implementation reading or writing the store can both encode and decode. Blosc(zstd-5, byte-shuffle) satisfies this on both `zarrs` and `zarr-python`.

### 3.6. Sharding

Sharding is off by default. It may be enabled per track via `shard_size` and `shard_column_size`. Under sharding the shard is the chunk at the Zarr `Array` level; writers write whole shards at a time (§5.4). The default may flip in a later revision once the sharding-sweep benchmark produces guidance.

---

## 4. Metadata Schema

PBZ metadata is split by scope: the store root carries a bare marker, and every track group carries its own conformance block. There is no store-level map of per-track metadata; each track is self-describing.

### 4.1. Root Marker

The root group's attributes carry only the `zarr_conventions` marker announcing that its subgroups may be PBZ tracks:

```json
{
  "zarr_conventions": [{ "uuid": "b7e3c1a2-...", "name": "perbase" }]
}
```

The root holds no genome, no contig list, and no track registry. Store discovery enumerates subgroups and treats a subgroup as a track only if it carries a complete conformance block (§4.3).

### 4.2. Track Conformance Block

A track group's `zarr.json` attributes carry the marker plus a `perbase:`-namespaced block. Fields here are the *interpretation* of the track; structural facts (dtype, chunk shape, dimension names) are read from the `values` array itself and are not repeated.

```json
{
  "zarr_conventions": [{ "uuid": "b7e3c1a2-...", "name": "perbase" }],
  "perbase:version": "0.4",
  "perbase:genome_checksum": "md5:...",
  "perbase:genome_name": "hg38",
  "perbase:ragged_index": "offsets",
  "perbase:ragged_contigs": "contigs",
  "perbase:coordinates": "0-based-half-open"
}
```

| Key                        | Required | Notes                                                                     |
|----------------------------|----------|---------------------------------------------------------------------------|
| `perbase:version`          | yes      | Format version. `0.4` at the time of writing.                             |
| `perbase:genome_checksum`  | yes      | `"md5:" + hex` over the canonical genome join (§2.1). Track identity.     |
| `perbase:genome_name`      | optional | Decorative assembly label (e.g., `hg38`). Excluded from the checksum.     |
| `perbase:ragged_index`     | yes      | Name of the offsets array in the group. Conventionally `"offsets"`.       |
| `perbase:ragged_contigs`   | yes      | Name of the contig-name array in the group. Conventionally `"contigs"`.   |
| `perbase:coordinates`      | yes      | Coordinate convention. Always `"0-based-half-open"`.                       |

Optional interpretation keys (`fill_value` semantics, `description`, `source`, and similar) are added as further `perbase:`-namespaced fields. Implementations MUST preserve unknown `perbase:` keys across read-modify-write cycles.

### 4.3. The Conformance Block Is the Completion Marker

The `perbase:` block is written **last**, as the final step of creating a track (ADR 0004). A `values` array is created at full shape and filled chunk-by-chunk in parallel; until the block lands, the group is either not a PBZ track or a crashed/partial import whose fill is ambiguous. Store discovery skips any subgroup without a complete block, so a partially written track never appears as complete. This commit-by-marker (rather than by directory rename) is chosen because a single small metadata write is the object-store-friendly primitive, keeping the design cloud-native ahead of remote-store work (§9).

---

## 5. I/O Conventions

### 5.1. Coordinates

All positional coordinates in PBZ, including region queries, write extents, and chunk boundaries, are **0-based and half-open**. A query `chr1:1000-2000` refers to positions `[1000, 2000)`, a 1000-base region. Conversion from 1-based formats (BED lines in some tools, VCF `POS`, etc.) is the caller's responsibility, done at the import boundary.

### 5.2. Region Queries

The canonical region-query form is `<contig>:<start>-<end>`. A region selects a contiguous slice on the position axis of a single contig, resolved to a flat span through the track's `offsets`. 2D tracks may be further sliced on the column axis by label (a single sample or a subset). A many-region query returns results keyed by each region's identity, in data (genomic) order, not input order.

### 5.3. Chunking and Write-Units

Default chunk shape is `(chunk_size, column_chunk_size)` for 2D tracks and `(chunk_size,)` for scalar tracks. The defaults of 1,000,000 positions × 16 columns are tuned for the human genome and typical cohort sizes; they are not load-bearing for correctness and may be overridden per track. The **write-unit** is the on-disk write granularity: a chunk, or a shard when the track is sharded. Import partitions work one task per write-unit, so concurrent writers own disjoint files.

### 5.4. Concurrent Writes

Concurrent writes to non-overlapping write-units of the same Zarr array are safe under both `zarrs` and `zarr-python`. PBZ import partitions tasks one per write-unit, landing each on the on-disk chunk (or shard) grid, so every task writes one whole unit that no other task touches. This keeps `zarrs` on its single-encode fast path and avoids any concurrent read-modify-write hazard. Under sharding, the shard is the chunk at the `Array` level and there is no sub-shard write API, so writers write whole shards; a sub-shard write would read-modify-write the entire shard.

### 5.5. Fill Semantics and Reductions

Region reductions (mean, sum, count, min, max, var, std) are **nan-aware**, and a position's participation is decided solely by the track's fill value, not special-cased per source format (ADR 0002):

* A `NaN` fill (as bigWig produces for uncovered positions) is skipped. A `mean` averages only covered positions; a `count` returns covered base pairs.
* A numeric fill participates as its value. For a d4 depth track, uncovered positions read back as `0`, indistinguishable from true zero depth, so they count: a `mean` is the sum over the full region length, and a `count` returns the region length.

PBZ does not add a separate missing-data channel to let a numeric-fill track distinguish "uncovered" from a true value. That distinction does not exist in the d4 source, so inventing it would fabricate information. The consequence is intended: the same rule gives each format the biologically correct default with no flag (a bigWig gap does not poison an exon mean; a d4 zero is not silently dropped), at the cost that `count` does not always mean region length.

---

## 6. Library Surfaces

The format is implementation-agnostic. The two libraries below are the reference implementations. They are deliberately independent at the format layer: the convention (checksum canonicalization, offset math, metadata serialization) is implemented once per language rather than shared, so the interop gates test two real writers rather than one writer checking itself (ADR 0001).

### 6.1. Rust (`pbzarr` crate)

The Rust workspace splits into three crates: `pbzarr` (the core convention library, the only crate published to crates.io), `pbzarr-readers` (input-format readers, which own git dependencies so the core stays publishable), and `pbzarr-python` (the PyO3 `_native` extension).

| Module / crate          | Responsibility                                                                |
|-------------------------|-------------------------------------------------------------------------------|
| `genome.rs`             | `Genome`, `Contig`, `ContigId`, `Region`, `offsets`, `checksum`. Pure types.  |
| `region_query.rs`       | `RegionQuery` and the `<contig>:<start>-<end>` parser.                         |
| `io/dtype.rs`           | `Dtype` tag, `Numeric` trait, dtype-to-Rust-type mapping.                      |
| `io/reader.rs`          | `ValueReader` trait. Import sources implement this.                            |
| `store.rs`              | `PbzStore`. A storage-agnostic container of track groups; holds no genome.     |
| `track.rs`              | `Track`, `TrackConfig`. Owns one `Genome`; caches its flat `values` handle.    |
| `stack.rs`              | `stack`. Combines N single-sample stores into one cohort store.                |
| `import/pipeline.rs`    | `run_pipeline`. crossbeam-channel work distribution across workers.            |
| `pbzarr-readers`        | `D4Reader` / `BigWigReader` / `BedReader` and their `from_*` entry points.     |
| `error.rs`              | `PbzError`, derived via `thiserror`.                                           |

`Track` is not generic over dtype. The on-disk dtype comes from metadata; runtime checks at `read_region<T>` and `write_region<T>` reject mismatches via `PbzError`. The trade-off is one runtime check per region I/O against simpler types for callers that mix tracks of different dtypes in one code path.

```rust
// Manual authoring path: state the genome explicitly.
let mut store = PbzStore::create("out.pbz")?;
let genome = Genome::from_fai("hg38.fa.fai")?.with_name("hg38");

store.create_track("mask", genome.clone(), TrackConfig::new(Dtype::Bool))?;
store.create_track(
    "depth",
    genome,
    TrackConfig::new(Dtype::I32)
        .columns(vec!["A".into(), "B".into(), "C".into()])
        .column_dim("sample"),
)?;

let track = store.track("depth").ok_or_else(/* ... */)?;
let region = track.genome().resolve(&"chr1:1000-2000".parse()?)?;
let data: ArrayD<i32> = track.read_region(&region)?;
```

`PbzStore::create(path)` / `open(path)` build a `FilesystemStore` and delegate to `create_with_storage` / `open_with_storage`, which take any synchronous `ReadableWritableListableStorage` trait object (filesystem or in-memory today; a future async-to-sync remote adapter slots in unchanged, ADR 0005). `create_track(name, genome, config)` takes the genome explicitly and is the manual path. The import entry points are the common path and create the track themselves (§6.2, ADR 0003).

**Import pipeline.** `pbzarr::import::run_pipeline<T, R: ValueReader>` parallelizes import at the write-unit level. It forks one reader per worker, partitions tasks one per write-unit across a bounded `crossbeam-channel`, and collects the first error into a shared `Mutex<Option<PbzError>>`. A task may straddle contig boundaries; the worker fills each overlapping contig's slice of the buffer by name before the single whole-unit write. The caller sets `Config::workers` to choose parallelism; the library spawns no pool of its own.

**Concurrency model.** Synchronous, with parallelism opt-in through the import pipeline. No `rayon`, no async runtime, no internal thread pools. Library code contains no `unwrap`, `expect`, or `panic!`; all failure modes surface as `PbzError`.

**Escape hatches.** Callers who need codec inspection, sharded subset reads, or custom chunk iteration drop down to the underlying `zarrs::Array` handle without losing the metadata layer above.

### 6.2. Python (`pbzarr`)

The Python wheel is a maturin mixed project. Python owns the entire convention layer natively over `zarr-python` and `numpy` (store and track creation, metadata read/write, `offsets`, `genome_checksum`, region-to-flat translation, region reads, and region-reduction planning). It delegates to the Rust `_native` extension only for the bulk format decoders and the multi-store stack, which release the GIL during the operation (ADR 0001).

The store is a bare container of tracks; the genome and the region API live on `Track`.

```python
import pbzarr

store = pbzarr.PbzStore.create("out.pbz")

# Import entry points create the track from the source headers: one source is a
# scalar (1D) track, several are a 2D track whose column labels come from
# the file stems (or explicit labels). All sources must share a genome_checksum.
store.track("mean_depth").import_d4([("sample.d4",)])
store.track("depth").import_d4([("A.d4", "A"), ("B.d4", "B"), ("C.d4", "C")])
store.track("signal").import_bigwig([("sample.bw",)])

# BED import takes an explicit genome (.fai / chrom.sizes), since BED carries no
# lengths. One named column per call, or every column in a single pass.
store.track("score").import_bed(
    [("a.bed.gz", "A")], column="score", dtype="float32", genome="hg38.fai"
)
store.import_bed_multi(
    "calls.bed.gz", {"score": "float32", "qual": "int32"}, genome="hg38.fai"
)
```

`stack` is the combine leg of the width spectrum: it reads N single-sample stores and writes one cohort store, turning each shared scalar track into a `(ΣL, N)` track. All sources must share a `genome_checksum`.

```python
cohort = pbzarr.stack(
    [("s1.pbz", "s1"), ("s2.pbz", "s2")], out="cohort.pbz", column_dim="sample"
)
```

### 6.3. Read API

Tracks are the read unit. Reads default to lazy/dask (chunks aligned to the on-disk grid); pass `chunks=None` for eager numpy.

```python
store = pbzarr.PbzStore("out.pbz")
store.tracks()                                     # ['depth', 'mask', ...]

depth = store.track("depth")
depth.region("chr1:1000-2000")                     # (position, sample) DataArray
depth.region("chr1:1000-2000", column="A")         # one sample
depth.region(["chr1:0-500", "chr2:10-20"])         # gather several regions

# Reduce many disjoint intervals into a (region x column) matrix. Reads only the
# chunks the intervals touch; runs eager (numpy) or lazy (dask), backed by flox.
depth.region_reduced([("chr1", 0, 500), ("chr2", 10, 20)], reduce="mean")

# Or open the whole store as a plain xarray DataTree.
dt = pbzarr.open("out.pbz")            # eager
dt = pbzarr.open("out.pbz", chunks={}) # dask-backed
```

Because a PBZ store is a normal Zarr v3 store, any Zarr reader (plain `zarr-python`, `xarray`) can read it directly with no PBZ-specific client.

---

## 7. Cross-Language Interop

The on-disk layout, metadata schema, dtype tags, dimension names, coord arrays, and codec configuration are all writer-agnostic. Concretely:

* **Variable-length UTF-8 string arrays** (Zarr v3 `vlen-utf8` codec) round-trip cleanly between `zarrs` (Rust) and `zarr-python` (Python). The per-track `contigs` and column-label arrays use this encoding.
* **Zarr v3 `dimension_names`** are written into array metadata directly, not as a separate `_ARRAY_DIMENSIONS` attribute. The xarray-zarr backend reads these without additional configuration.
* **Codec pipelines** live in each array's own `zarr.json` and are fixed at creation; a later writer filling an existing array encodes through the recorded pipeline. A cross-language store must therefore use only codecs every implementation can both encode and decode; Blosc(zstd-5, byte-shuffle) qualifies.
* **The `genome_checksum`** is computed independently by each implementation over the same canonicalization (`"{name}\t{length}\n"`, sorted by name). Because the two implementations do not share code (ADR 0001), a matching checksum is a real interop signal rather than a tautology; a pinned canonicalization plus a golden-vector test keep them byte-identical.
* **The `perbase:` conformance block** is preserved verbatim by both implementations; unknown keys survive a read-write cycle.

Cross-language round-trip is enforced by a release gate that writes a fixture store from one language and reads it from the other.

---

## 8. Design Considerations

**Two independent implementations, one spec.** The Python wheel bundles the Rust `_native` extension, which makes it tempting to delegate all format-layer work to Rust and maintain it once. PBZ deliberately does not: Python owns its convention layer natively, and DRY lives in the format spec rather than shared code (ADR 0001). The cost is that offset math and checksum canonicalization exist twice; the payoff is that the interop gates prove two real writers agree, and the Python read path returns native xarray/dask/zarr objects. The only future candidate for crossing into Rust is the region-reduction inner loop, and only if profiling shows it dominates.

**Storage-agnostic core.** `PbzStore` and `Track` hold a synchronous storage trait object, not a concrete filesystem store (ADR 0005). Filesystem and in-memory backends work today; a `MemoryStore` round-trips the full API, which demonstrates that the deferred async-to-sync remote adapter (same sync traits) will slot in without a core change. The flat layout (few large objects, disjoint write-unit objects) plus the completion-marker commit is already the object-store write pattern, so remote write is a backend addition, not a redesign.

---

## 9. Open Questions and Future Work

* **Append modes.** Appending samples (columns) or contigs to an existing store is not supported today. The planned approach stages new columns as a cheap size-1 tail on a rectilinear column grid, compacted into wide chunks later, so appends never rewrite existing chunks. It is blocked on rectilinear chunk-grid support reaching a released xarray; zarr-python supports it in a released version, and the zarrs writer lands in an unreleased version, so the Rust side waits on that release.
* **Cloud writes.** Synchronous I/O against local files is the current target, and remote read over `HTTPStore` can be wired up now. Cloud writes are deferred to a feature-gated `object_store`/`opendal` adapter that encapsulates a runtime; tokio enters only that feature's build, and the core (already consuming the sync trait object) needs no change when it lands (ADR 0005).
* **Validation helpers.** A `validate` helper that audits a store for spec conformance (coord-array consistency, dtype tags, checksum agreement) is on the list; readers currently trust the writer.
* **Additional import formats.** d4, bigWig, and BED are the built-in importers today. bedGraph and BAM/CRAM are the obvious additions; each needs a `ValueReader` and a `from_*` entry point in `pbzarr-readers`. A multi-column BED into a single wide 2D track (rather than one scalar track per column) is also planned.
* **Multi-resolution tracks.** Coarsened views would nest further under the track group; not yet designed.
* **Command-line interface.** A `pbz` CLI wrapping import, inspection, and text export is deferred until the library API has absorbed downstream feedback.

---

## 10. Conclusion

PBZ is a thin convention on top of Zarr v3 that aligns per-base genomic data along the axis on which it is most often computed: position × sample, with samples co-located in compressed chunks. It is deliberately small in surface area, comprising a flat self-describing track layout, a per-track metadata schema, and a set of I/O conventions. The reference implementations demonstrate that the convention is round-trippable across languages and tractable to build against existing Zarr toolchains. The format's value will be determined empirically by the cohort-compression, mask-generation, and cross-sample-math benchmarks; this document captures the design those benchmarks will validate.
