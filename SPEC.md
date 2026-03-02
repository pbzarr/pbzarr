# PBZ (Per-Base Zarr) Specification

**Version 0.1 — DRAFT**

## 1. Overview

PBZ is a Zarr v3 based format for storing per-base resolution genomic data. It provides efficient storage and querying of data like read depth, methylation levels, accessibility scores, and boolean masks across multiple samples, cells, or other column types.

PBZ is designed as a modern alternative to formats like D4 and bigWig, leveraging the Zarr ecosystem for compression, chunking, and integration with Python, Dask, and Xarray.

PBZ is a **convention and domain layer** on top of Zarr, not an array library. It defines a standard layout for genomic data within a Zarr v3 store, provides region parsing and column resolution, and delegates all array storage, compression, and I/O to zarr-python.

### 1.1 Design Goals

1. Fast cross-column queries at genomic regions (the common case)
2. Efficient single-column queries
3. Reasonable cost to add new columns
4. Good compression for both dense and sparse data
5. Simple, predictable structure
6. Self-describing tracks with independent schemas
7. Thin layer over Zarr — delegate storage, compression, and I/O to zarr-python

### 1.2 File Extension

PBZ stores use the `.pbz.zarr` extension (e.g., `depths.pbz.zarr`).

### 1.3 Conventions

The key words "MUST", "MUST NOT", "SHOULD", "SHOULD NOT", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

## 2. Store Structure

```
my_data.pbz.zarr/
├── zarr.json                     # Root metadata
├── contigs                       # Array: contig names
├── contig_lengths                # Array: contig lengths
└── tracks/
    ├── {track_name}/
    │   ├── zarr.json             # Track metadata
    │   ├── columns               # Array: column names (optional)
    │   ├── {contig_name}/        # Data array for each contig
    │   └── ...
    └── {group}/{track_name}/     # Tracks can be nested in groups
        └── ...
```

### 2.1 Root Metadata

The root Zarr group attributes (`/zarr.json`) MUST contain:

```json
{
  "perbase_zarr": {
    "version": "0.1"
  }
}
```

| Field   | Type   | Required | Description                          |
|---------|--------|----------|--------------------------------------|
| version | string | Yes      | Per-Base Zarr specification version  |

### 2.2 Contig Arrays

**`/contigs`**

| Property    | Value                  |
|-------------|------------------------|
| Shape       | `(num_contigs,)`       |
| Dtype       | Variable-length string |
| Description | Contig/chromosome names |

**`/contig_lengths`**

| Property    | Value                              |
|-------------|------------------------------------|
| Shape       | `(num_contigs,)`                   |
| Dtype       | int64                              |
| Description | Length of each contig in base pairs |

The order of contigs MUST be consistent between these two arrays.

## 3. Tracks

A track is a named group containing per-base data for all contigs. Tracks are stored under `/tracks/` and MAY be nested in subgroups (e.g., `/tracks/masks/callable/`).

### 3.1 Track Metadata

Track group attributes (`/tracks/{name}/zarr.json`) MUST contain:

```json
{
  "perbase_zarr_track": {
    "dtype": "uint32",
    "chunk_size": 1000000,
    "column_chunk_size": 16,
    "has_columns": true,
    "description": "Read depth from BAM files",
    "source": "perbase v0.1.0"
  }
}
```

| Field             | Type    | Required       | Description                                                                 |
|-------------------|---------|----------------|-----------------------------------------------------------------------------|
| dtype             | string  | Yes            | One of: `uint8`, `uint16`, `uint32`, `int8`, `int16`, `int32`, `float32`, `float64`, `bool` |
| chunk_size        | integer | Yes            | Chunk size along position axis (bp)                                         |
| column_chunk_size | integer | If has_columns | Chunk size along column axis                                                |
| has_columns       | boolean | Yes            | Whether this track has a column dimension                                   |
| description       | string  | No             | Human-readable description                                                  |
| source            | string  | No             | Tool/version that created this track                                        |

Additional tool-specific metadata MAY be included in the `perbase_zarr_track` object.

### 3.2 Columns Array

Present only if `has_columns` is `true`.

**`/tracks/{name}/columns`**

| Property    | Value                                            |
|-------------|--------------------------------------------------|
| Shape       | `(num_columns,)`                                 |
| Dtype       | Variable-length string                           |
| Description | Column identifiers (sample names, cell IDs, etc.) |

### 3.3 Data Arrays

One array per contig, named by contig name (`/tracks/{name}/{contig}/`).

**With columns** (`has_columns: true`):

| Property | Value                            |
|----------|----------------------------------|
| Shape    | `(contig_length, num_columns)`   |
| Chunks   | `(chunk_size, column_chunk_size)` |
| Dtype    | As specified in track metadata   |

**Without columns** (`has_columns: false`):

| Property | Value                          |
|----------|--------------------------------|
| Shape    | `(contig_length,)`             |
| Chunks   | `(chunk_size,)`                |
| Dtype    | As specified in track metadata |

### 3.4 Data Array Attributes

Each data array SHOULD include dimension names following Xarray conventions:

```json
{ "_ARRAY_DIMENSIONS": ["position", "column"] }
```

Or for tracks without columns:

```json
{ "_ARRAY_DIMENSIONS": ["position"] }
```

## 4. Chunking and Sharding

PBZ uses 2D chunking for tracks with columns to balance query performance and append efficiency.

### 4.1 Recommended Chunk Defaults

| Setting            | Default   | Description          |
|--------------------|-----------|----------------------|
| chunk_size         | 1,000,000 | Positions per chunk  |
| column_chunk_size  | 16        | Columns per chunk    |

**Rationale.** Large position chunks provide efficient sequential access and good compression. Small column chunks enable parallel decompression, limit the cost of appending columns (only edge chunks are rewritten), and improve cache utilization for cross-column queries.

### 4.2 Chunk Size Guidelines

| Use Case                    | chunk_size    | column_chunk_size |
|-----------------------------|---------------|-------------------|
| General purpose             | 1,000,000     | 16                |
| Many columns (>1000)        | 1,000,000     | 32–64             |
| Frequent column additions   | 1,000,000     | 8–16              |
| Small contigs (<1 Mbp)      | contig_length | 16                |

### 4.3 Sharding

Zarr v3 supports **sharding**, which stores multiple chunks within a single storage object (file). Within a shard, chunks are compressed and serialized independently, so individual chunks can still be read without decompressing the entire shard. However, writes are most efficient at shard granularity.

Sharding is OPTIONAL in PBZ. Implementations SHOULD support it by passing through zarr-python's `shards` parameter in track creation.

**When to use sharding.** Without sharding, each chunk is a separate file. For a 250 Mbp chromosome with 1M position chunks and a columnar track, that's 250 files per contig per track. On local filesystems this is fine. On object stores (S3, GCS) or stores with many tracks and contigs, the file count can become a performance problem. Sharding reduces file count while preserving fine-grained read access.

**Recommended shard configuration:**

| Scenario | chunk_size | shard_size (positions) | Chunks per shard | Files per 250 Mbp contig |
|---|---|---|---|---|
| No sharding (default) | 1,000,000 | — | 1 | 250 |
| Moderate sharding | 1,000,000 | 50,000,000 | 50 | 5 |
| Aggressive sharding | 1,000,000 | 250,000,000 | 250 | 1 |

The shard shape applies to both dimensions. For a track with columns, a shard shape of `(50_000_000, column_chunk_size)` means each shard file contains 50 position chunks × 1 column chunk.

**Tradeoffs:**

- Sharding reduces file count and can improve performance on object stores and high-latency filesystems.
- Sharding increases write amplification — writing a single chunk may require rewriting the entire shard.
- Without sharding, each chunk can be written independently, which is better for parallel or incremental writes.

If `shard_size` is specified in track metadata, implementations MUST record it:

```json
{
  "perbase_zarr_track": {
    "dtype": "uint32",
    "chunk_size": 1000000,
    "column_chunk_size": 16,
    "shard_size": 50000000,
    "shard_column_size": 16,
    "has_columns": true
  }
}
```

| Field              | Type    | Required      | Description                          |
|--------------------|---------|---------------|--------------------------------------|
| shard_size         | integer | No            | Shard size along position axis (bp)  |
| shard_column_size  | integer | If sharded and has_columns | Shard size along column axis |

If shard fields are absent, the track is not sharded.

## 5. Compression

PBZ recommends Blosc compression with the following settings:

| Setting    | Value        |
|------------|--------------|
| Compressor | Zstd         |
| Level      | 5            |
| Shuffle    | Byte shuffle |

Implementations MAY use any Zarr-compatible compression codec. Compression configuration is handled entirely by zarr-python's codec system — PBZ does not define its own compression abstraction.

## 6. Coordinate System

1. Positions are **0-based, half-open** (like BED format).
2. Position `i` in an array corresponds to reference base pair `[i, i+1)`.
3. Contig names MUST match the reference genome exactly.

## 7. Missing Data

PBZ delegates missing data handling to Zarr's `fill_value` mechanism. The `fill_value` is set at array creation time and is returned for any chunk that has never been written.

Implementations SHOULD use the following default fill values when creating data arrays:

| Dtype Category | Default Fill Value | Rationale                        |
|----------------|--------------------|----------------------------------|
| Integer types  | `0`                | Zarr default; natural for counts |
| Float types    | `NaN`              | Standard missing indicator       |
| Bool           | `False`            | Zarr default                     |

Implementations MAY allow users to override the fill value at track creation time by passing it through to zarr-python's array creation.

PBZ does not define its own sentinel values or missing data abstraction. Users who need explicit missing value semantics (e.g., distinguishing "zero depth" from "no data") SHOULD use float dtypes where `NaN` provides a natural missing indicator, or manage sentinel conventions in their own application layer.

## 8. Genomic Region Syntax

PBZ defines a standard region query interface. All coordinates are 0-based, half-open, consistent with the PBZ coordinate system (§6).

### 8.1 String Form

```
contig
contig:start-end
contig:position
```

| Pattern            | Example          | Meaning                                         |
|--------------------|------------------|-------------------------------------------------|
| `contig`           | `chr1`           | Entire contig, equivalent to `chr1:0-{length}`  |
| `contig:start-end` | `chr1:1000-2000` | Positions `[1000, 2000)`                        |
| `contig:position`  | `chr1:5000`      | Single base at position 5000, equivalent to `chr1:5000-5001` |

Whitespace around components is stripped. Commas in numeric values are stripped (e.g., `chr1:1,000,000-2,000,000` is valid). Contig names MAY contain alphanumeric characters, underscores, hyphens, and periods. The colon (`:`) is reserved as the contig/coordinate delimiter.

### 8.2 Tuple Form

```python
(contig,)
(contig, start, end)
(contig, position)
```

| Pattern                | Example              | Meaning                    |
|------------------------|----------------------|----------------------------|
| `(contig,)`            | `("chr1",)`          | Entire contig              |
| `(contig, start, end)` | `("chr1", 1000, 2000)` | Positions `[1000, 2000)` |
| `(contig, position)`   | `("chr1", 5000)`     | Single base at position 5000 |

`start` and `end` MUST be non-negative integers. `start` MUST be less than `end`.

### 8.3 Slice Form

Python implementations SHOULD support `__getitem__` with this indexing convention:

```python
track[contig, position_slice]              # without columns
track[contig, position_slice, column_sel]  # with columns
```

Where:

- `contig` — a string contig name
- `position_slice` — a Python `slice`, `int`, or `Ellipsis` (`...` for all positions)
- `column_sel` — a `slice`, `int`, `str`, `list[str]`, or `Ellipsis`

Examples:

```python
track["chr1", 1000:2000, :]                 # region, all columns
track["chr1", 1000:2000, 0:10]              # region, first 10 columns by index
track["chr1", 1000:2000, "sample_A"]        # region, single column by name
track["chr1", 1000:2000, ["sample_A", "B"]] # region, multiple columns by name
track["chr1", ...]                          # entire contig, all columns
track["chr1", 5000]                         # single position, all columns
```

### 8.4 One-Based Coordinate Support

Implementations SHOULD accept an optional `one_based` flag (default `False`) on query methods. When `one_based=True`, coordinates are interpreted as 1-based inclusive (matching samtools/bcftools conventions) and converted internally to 0-based half-open:

- `chr1:1000-2000` with `one_based=True` → 0-based half-open `[999, 2000)`

Implementations SHOULD document prominently when `one_based` is in use to avoid silent coordinate errors.

### 8.5 Region Parsing

Implementations MUST provide a `parse_region()` function:

```python
parse_region(region: str | tuple, *, one_based: bool = False) -> Region
```

The returned `Region` object MUST always use 0-based, half-open coordinates, regardless of the `one_based` input flag:

```python
@dataclass
class Region:
    contig: str
    start: int | None   # None means 0 (start of contig)
    end: int | None      # None means contig length
```

### 8.6 Validation

Implementations MUST raise an error when:

- `contig` is not present in the store
- `start >= end` (empty or inverted range)
- `start < 0` or `end < 0`
- `end > contig_length` (out of bounds)
- The region string is malformed (e.g., `chr1:abc-def`)
- A column name in `columns` is not present in the track

## 9. Query Method

Implementations MUST provide a `query()` method on track objects:

```python
track.query(region, *, columns=None, one_based=False) -> numpy.ndarray | dask.array.Array
```

| Parameter  | Type                       | Default  | Description                                                      |
|------------|----------------------------|----------|------------------------------------------------------------------|
| region     | `str \| tuple`             | required | Region in string form (§8.1) or tuple form (§8.2)               |
| columns    | `str \| list[str] \| None` | `None`   | Column filter. `None` returns all columns.                       |
| one_based  | `bool`                     | `False`  | If `True`, interpret coordinates as 1-based inclusive (§8.4).    |

**Return value.** The return type depends on the backend (§10). The shape is `(length,)` for tracks without columns, or `(length, num_selected_columns)` for tracks with columns.

**Implementation.** The `query()` method is responsible for region parsing (§8), column name resolution, and validation (§8.6). Once the target contig, position slice, and column slice are resolved, the method delegates to the underlying `zarr.Array` for data access. PBZ does not reimplement array slicing or compression — it constructs the correct slice and calls through to zarr-python.

### 9.1 Multi-Region Queries

Implementations MAY support querying multiple regions at once:

```python
track.query_regions(["chr1:1000-2000", "chr2:5000-6000"])
```

The return type for multi-region queries is implementation-defined but SHOULD be a list of arrays or a dictionary keyed by region string.

## 10. Backend Model

PBZ supports multiple array backends. The backend is selected at store open time and determines the return type for all data access operations on that store. The backend is a **thin dispatch layer** — PBZ resolves the region, columns, and validation, then makes a single call to zarr-python (numpy backend) or dask (dask backend) at the end.

### 10.1 Opening a Store

```python
store = pbz.open("data.pbz.zarr")                  # default: numpy
store = pbz.open("data.pbz.zarr", backend="numpy")  # explicit
store = pbz.open("data.pbz.zarr", backend="dask")   # lazy/chunked
```

### 10.2 Supported Backends

| Backend   | Return Type         | Eager/Lazy | Required Dependency |
|-----------|---------------------|------------|---------------------|
| `"numpy"` | `numpy.ndarray`     | Eager      | numpy (core)        |
| `"dask"`  | `dask.array.Array`  | Lazy       | dask (optional)     |

`"numpy"` is the default. Implementations MUST support `"numpy"`. Support for `"dask"` is RECOMMENDED.

### 10.3 Behavior

1. The backend is stored on the store instance. Tracks inherit the backend from their parent store.
2. Query and region parsing logic MUST be identical across backends. The backend only affects the final data access call.
3. For `"numpy"`: the resolved slice is passed to `zarr.Array.__getitem__()`, returning an eager numpy array.
4. For `"dask"`: the resolved slice is applied to `dask.array.from_zarr()` on the underlying zarr array, returning a lazy dask array. The dask chunk structure inherits from the zarr store's chunk configuration.

### 10.4 Chunk Size Implications for Dask

When using the `"dask"` backend, PBZ's chunk configuration directly determines the dask task graph structure:

- `chunk_size` (position axis) controls how many base pairs are in each dask task.
- `column_chunk_size` controls how many columns are in each dask task.

The default chunking (1,000,000 positions × 16 columns) provides good parallelism for typical genomic region queries. Users tuning PBZ chunk sizes SHOULD be aware they are also tuning dask performance.

### 10.5 Backend Extensibility

The set of supported backends is open. Implementations SHOULD design the backend dispatch such that adding a new backend requires only a new last-mile function (array creation + slicing), not changes to region parsing, column resolution, or validation.

Future backends MAY include:

| Backend   | Return Type          | Use Case                        |
|-----------|----------------------|---------------------------------|
| `"cupy"`  | `cupy.ndarray`       | GPU arrays via `cupy.asarray()` |
| `"jax"`   | `jax.Array`          | JAX ecosystem / XLA devices     |

GPU-native I/O (bypassing CPU entirely) is expected to come through the **store layer** rather than the backend layer. For example, RAPIDS kvikio provides a zarr-compatible store that reads directly into GPU memory via GPUDirect Storage. Since PBZ passes the store through to zarr-python (§11.2), this requires no PBZ changes:

```python
import kvikio.zarr
store = pbz.open("data.pbz.zarr", store=kvikio_gds_store)
```

When adding future backends, implementations SHOULD:

- Ensure `query()` does not perform `isinstance` checks against `numpy.ndarray` or any specific array type on the return path.
- Ensure returned arrays are compatible with the [Python Array API Standard](https://data-apis.org/array-api/latest/) where possible, so downstream code can consume PBZ output using `x.__array_namespace__()` regardless of backend.
- Keep backends as optional dependencies with lazy imports, following the same pattern as dask (§10.6).

### 10.6 Optional Dependencies

Dask is an **optional dependency**. Implementations MUST NOT require dask for core functionality. If a user requests `backend="dask"` without dask installed, implementations MUST raise an `ImportError` with a clear installation instruction.

Implementations SHOULD declare optional dependencies:

```toml
[project.optional-dependencies]
dask = ["dask[array]"]
xarray = ["xarray", "dask[array]"]
all = ["dask[array]", "xarray"]
```

### 10.7 Xarray Integration

Implementations SHOULD provide a `to_xarray()` method on tracks:

```python
ds = track.to_xarray()                           # entire track
ds = track.to_xarray(region="chr1:1000-2000")    # specific region
```

This returns an `xarray.DataArray` backed by dask arrays, with dimensions labeled according to the `_ARRAY_DIMENSIONS` attribute (`"position"`, `"column"`). The `to_xarray()` method requires both xarray and dask to be installed.

This method exists because PBZ owns the dimension labeling and column naming that xarray needs — constructing a properly-labeled `DataArray` from a raw zarr array requires knowledge of the PBZ layout that the user shouldn't have to reconstruct.

## 11. Escape Hatches

PBZ is a convention layer on top of zarr-python. Users SHOULD be able to drop down to raw zarr objects at any point for operations that PBZ does not anticipate.

### 11.1 Raw Zarr Access

Implementations MUST provide access to the underlying zarr objects:

```python
store = pbz.open("data.pbz.zarr")

# Raw zarr.Group for the store root
store.root  # zarr.Group

# Raw zarr.Array for a specific contig in a track
track = store["depths"]
arr = track.zarr_array("chr1")  # zarr.Array, shape (contig_length, num_columns)
```

This allows users to use any zarr-python feature directly — advanced indexing, block selection, `info_complete()`, direct `dask.array.from_zarr()` calls, or any future zarr functionality — without PBZ needing to wrap or expose it.

### 11.2 Storage Backend Passthrough

PBZ does not define its own storage abstraction. The `store` parameter in `pbz.open()` and `pbz.create()` is passed through to `zarr.open_group()` / `zarr.create_group()`. Any zarr-compatible store (local filesystem, S3 via fsspec, `MemoryStore`, `ObjectStore`, etc.) works transparently:

```python
# Local filesystem (default)
store = pbz.open("data.pbz.zarr")

# S3
store = pbz.open("s3://bucket/data.pbz.zarr", storage_options={"anon": True})

# In-memory
store = pbz.create({}, contigs=["chr1"], contig_lengths=[248956422])
```

### 11.3 Compression Passthrough

PBZ recommends default compression settings (§5) but does not wrap zarr's codec system. Implementations SHOULD accept zarr codec objects directly in track creation:

```python
import zarr

store.create_track("depths",
    dtype="uint32",
    columns=["sample_A"],
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=9))
```

## 12. Operations

### 12.1 Reading Data

```python
store = pbz.open("data.pbz.zarr")

# List tracks
store.tracks()  # ["depths", "masks/callable"]

# Get a track
depths = store["depths"]

# Query with region syntax
data = depths.query("chr1:1000000-2000000")                  # all columns
data = depths.query("chr1:1000000-2000000", columns=["sample_A"])

# Query with slice syntax
data = depths["chr1", 1000000:2000000, :]       # all columns
data = depths["chr1", 1000000:2000000, 0:10]    # first 10 columns

# Track info
depths.columns      # ["sample_A", "sample_B", ...]
depths.dtype        # uint32
depths.has_columns  # True

# Escape hatch to raw zarr
arr = depths.zarr_array("chr1")  # zarr.Array
```

### 12.2 Writing Data

```python
# Create store
store = pbz.create("output.pbz.zarr",
    contigs=["chr1", "chr2"],
    contig_lengths=[248956422, 242193529])

# Create track
store.create_track("depths",
    dtype="uint32",
    columns=["sample_A", "sample_B"])

# Create track with sharding (for object stores or large datasets)
store.create_track("depths",
    dtype="uint32",
    columns=["sample_A", "sample_B"],
    shard_size=50_000_000)

# Write data
store["depths"]["chr1", 0:1000000, :] = data_array
```

### 12.3 Adding a Track to an Existing Store

```python
store = pbz.open("data.pbz.zarr", mode="r+")

store.create_track("masks/callable",
    dtype="bool",
    columns=store["depths"].columns,
    metadata={"source": "clam loci", "min_depth": 10})

store["masks/callable"]["chr1"] = mask_array
```

### 12.4 Dask Backend

```python
store = pbz.open("data.pbz.zarr", backend="dask")

# All queries return dask arrays
data = store["depths"].query("chr1:1000000-2000000")
# → dask.array.Array, lazy, not yet computed

# Compute when ready
result = data.mean(axis=1).compute()
```

### 12.5 Xarray Integration

```python
store = pbz.open("data.pbz.zarr")

# Get a labeled DataArray (dask-backed)
ds = store["depths"].to_xarray()
# → xr.DataArray with dims ("position", "column"), dask-backed

# With region filter
ds = store["depths"].to_xarray(region="chr1:1000000-2000000")
```

## 13. Examples

### 13.1 Single Track Store (Depth Data)

```
depths.pbz.zarr/
├── zarr.json                     # {"perbase_zarr": {"version": "0.1"}}
├── contigs                       # ["chr1", "chr2"]
├── contig_lengths                # [248956422, 242193529]
└── tracks/
    └── depths/
        ├── zarr.json             # {"perbase_zarr_track": {"dtype": "uint32", "chunk_size": 1000000,
        │                         #                         "column_chunk_size": 16, "has_columns": true}}
        ├── columns               # ["sample_A", "sample_B", "sample_C"]
        ├── chr1/                 # shape: (248956422, 3)
        └── chr2/                 # shape: (242193529, 3)
```

### 13.2 Multi-Track Store (Depths + Masks)

```
analysis.pbz.zarr/
├── zarr.json
├── contigs
├── contig_lengths
└── tracks/
    ├── depths/
    │   ├── zarr.json             # dtype: uint32, has_columns: true
    │   ├── columns               # ["sample_A", "sample_B", "sample_C"]
    │   ├── chr1/
    │   └── chr2/
    ├── masks/
    │   └── callable/
    │       ├── zarr.json         # dtype: bool, has_columns: true
    │       ├── columns           # ["sample_A", "sample_B", "sample_C"]
    │       ├── chr1/
    │       └── chr2/
    └── stats/
        └── mean_depth/
            ├── zarr.json         # dtype: float32, has_columns: false
            ├── chr1/             # shape: (248956422,) — no column dimension
            └── chr2/
```

### 13.3 Track with Different Columns (Per-Population Mask)

```
└── tracks/
    ├── depths/
    │   ├── columns               # ["sample_A", "sample_B", "sample_C"]
    │   └── ...
    └── masks/
        └── callable_by_pop/
            ├── zarr.json         # {"perbase_zarr_track": {..., "population_map": {
            │                     #     "pop_A": ["sample_A", "sample_B"],
            │                     #     "pop_B": ["sample_C"]}}}
            ├── columns           # ["pop_A", "pop_B"]
            ├── chr1/             # shape: (length, 2)
            └── chr2/
```

### 13.4 Sharded Store (Object Store Deployment)

```
depths.pbz.zarr/
├── zarr.json                     # {"perbase_zarr": {"version": "0.1"}}
├── contigs                       # ["chr1"]
├── contig_lengths                # [248956422]
└── tracks/
    └── depths/
        ├── zarr.json             # {"perbase_zarr_track": {"dtype": "uint32", "chunk_size": 1000000,
        │                         #                         "column_chunk_size": 16, "shard_size": 50000000,
        │                         #                         "shard_column_size": 16, "has_columns": true}}
        ├── columns               # ["sample_A", "sample_B", "sample_C"]
        └── chr1/                 # 5 shard files instead of 249 chunk files
```

## 14. Compatibility

### 14.1 Zarr Version

PBZ requires **Zarr v3** format.

### 14.2 Zarr-Python Version

PBZ implementations in Python SHOULD target **zarr-python ≥ 3.0** and use the v3 API (`zarr.create_array`, `zarr.open_group`, codec classes, etc.).

### 14.3 Dimension Names

Arrays SHOULD include the `_ARRAY_DIMENSIONS` attribute following Xarray conventions for interoperability (see §3.4).

## 15. Future Considerations

The following are not part of v0.1 but may be added in future versions:

- Additional array backends (`"cupy"`, `"jax"`) following the extensibility model in §10.5
- Formal adoption of the [Python Array API Standard](https://data-apis.org/array-api/latest/) for backend-agnostic downstream code
- GPU-native I/O via kvikio / GPUDirect Storage at the store layer
- Additional dimensions beyond position and columns
- Sparse data optimization hints
- Region indexing for fast range queries

## 16. Changelog

### Version 0.1

- Initial specification

## 17. Acknowledgments

This specification was informed by:

- [VCF Zarr Specification](https://github.com/pystatgen/vcf-zarr-spec)
- [D4 Format](https://github.com/38/d4-format)
- [Zarr v3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html)
- [bio2zarr](https://github.com/sgkit-dev/bio2zarr)