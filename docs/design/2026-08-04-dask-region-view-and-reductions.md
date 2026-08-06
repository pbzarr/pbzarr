# Python `region()`, `regions()`, and segmented reductions

Status: approved; current analytical design
Date: 2026-08-04

Depends on
`docs/superpowers/specs/2026-08-04-xarray-native-python-api-design.md`.
Supersedes the public interfaces in the July region/peak-store proposals. Reading
and writing an on-disk region representation remain deferred.

## Summary

```python
tree = pbzarr.open("cohort.pbz", tracks=["coverage", "fire_coverage"])

window = tree["coverage"].to_dataset().pbz.region("chr1:100-200")
regions = tree.pbz.regions(peaks)
summary = regions.pbz.reduce("mean")
```

`region` returns one per-base `xr.DataArray`. `regions` returns a derived packed
per-base `xr.Dataset`. `reduce` replaces its packed position segments with region.
There is no public read wrapper, planner, packed-layout, or result class.

Region work consumes already-open xarray arrays. It never reopens paths or copies
credentials into attrs. Regular/sharded opening is supported now; unequal in-memory
Dask chunks test general chunk reasoning without promising rectilinear on-disk
support.

## Scope

Goals:

- ordinary labeled-array results;
- planning proportional to regions/pieces/batches, never genome length;
- complete-region output batches and exact non-position Dask chunks;
- touched-block-only Dask graphs that share one source-block key across dependent
  batches;
- selected-piece-only eager reads for `chunks=None`, with the full result required
  to fit RAM; and
- xarray-owned numerical semantics and explicit large-result computation.

Non-goals:

- overlapping intervals or padded `region × relative_position` arrays;
- custom reducer callables or PBZ convenience reduction methods;
- `top`/summit selection;
- automatic Parquet/Polars output or scheduler/cluster setup;
- every possible length-preserving mutation detector; and
- any on-disk region schema.

## Public interfaces

### One query

`RegionQuery(contig, start, stop)` remains the public typed 0-based half-open
interval. `stop` is canonical throughout the parser, scalar tuple forms,
provenance, and `RegionLayout`; there is no pre-1.0 `end` field shim. Strings and
tuples normalize through one private `_normalize_one` function. Dataset access is:

```python
def region(self, query, *, column: str | None = None) -> xr.DataArray: ...
```

The accessor validates one normal track, resolves the query through eager
`contigs`/`offsets`, and directly slices `ds["values"]`. It may select one unique
label on the declared generic column dimension. The result retains an unindexed
position dimension and adds scalar `region_contig`, `region_start`, and
`region_stop`.

Queries never clamp. Unknown contigs, negative/out-of-range bounds, and
empty/reversed intervals are errors. Default Dask and `chunks=None` both remain
lazy at construction for this single native xarray slice.

### Many queries

```python
def regions(self, intervals) -> xr.Dataset: ...

# DataTree accessor
def regions(self, intervals, *, tracks=None) -> xr.Dataset: ...
```

Dataset input is one normal track or an exactly composed same-genome Dataset.
DataTree input composes already-open selected children through
`tree.pbz.dataset(tracks)` and delegates; it never fetches omitted children.

Interval input precedence is:

1. one `RegionQuery` or region string;
2. one scalar `(contig, start, stop)` tuple;
3. DataFrame-like recognized columns;
4. three equal-length 1D column arrays; and
5. iterable rows.

A DataFrame-like input has `.columns` and `__getitem__`. It must contain exactly one
contig alias from `contig`, `chrom`, or `#chrom`, exactly one `start`, and exactly
one endpoint alias from `stop` or `end`. Missing, duplicate, or ambiguous required
aliases are errors; unrelated columns are ignored. Selected 1D columns use `.to_numpy()` when
available, otherwise `np.asarray`. No pandas- or Polars-specific adapter exists.

Coordinates must be signed-64 integral scalars; booleans, floating/nonfinite
values, coercive objects, and overflow are rejected. Intervals must be nonempty,
in bounds, and pairwise disjoint. Adjacency is valid.

### Reduce packed position

```python
def reduce(self, reducer: str, /, **kwargs) -> xr.Dataset: ...
```

Allowed reducers are `mean`, `sum`, `min`, `max`, `count`, `std`, `var`, `median`,
and `quantile`. `dim` is forbidden because PBZ selects segmented position. Other
arguments go to xarray. Ordinary non-position operations may occur before or after:

```python
regions.mean("sample").pbz.reduce("median")
regions.pbz.reduce("median").mean("sample")
```

Order remains mathematically visible for nonlinear operations.

## Packed Dataset contract

```text
Dimensions:
    position: 10
    region: 3
    region_boundary: 4
    sample: 76

Coordinates:
    offsets(region_boundary)       = [0, 3, 7, 10]
    region_contig(region)          = ["chr1", "chr1", "chr2"]
    region_start(region)           = [100, 400, 20]
    region_stop(region)            = [103, 404, 23]
    region_input_index(region)     = [2, 0, 1]
    region_storage_index(region)   = [0, 1, 2]
    sample(sample)                 = ["s1", ..., "s76"]

Data variables:
    coverage(position, sample)
    fire_coverage(position, sample)
```

`position`, `region`, and `region_boundary` are unindexed. A standalone track keeps
the name `values`. There is no per-position region ID, contig, genomic position, or
ordinal coordinate.

Intervals use stable genomic storage order. `region_input_index` is the caller join
key; `region_storage_index == arange(n_regions)` binds provenance to segmentation.
Sorting a packed Dataset by input index is invalid because it reorders region-sized
coordinates without position rows. Reduce first, then sort the summary.

Derived attrs are:

```python
{
    "pbz:representation": "packed-regions",
    "pbz:parent_genome_checksum": "md5:...",
    "pbz:coordinates": "0-based-half-open",
}
```

They identify an in-memory contract, not an on-disk PBZ kind.

## Domain values and planning arrays

`RegionQuery` and `RegionLayout` are the only custom domain values.

```python
@dataclass(frozen=True)
class RegionLayout:
    contig_ids: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    flat_starts: np.ndarray
    flat_stops: np.ndarray
    packed_offsets: np.ndarray
    input_index: np.ndarray
```

Every stored ndarray is marked read-only. Normalization stable-sorts intervals by
genomic position and checks overlap vectorially.

There is no source or variable-plan class. The private planner consumes an xarray
DataArray directly:

```python
def _plan_variable_regions(
    values: xr.DataArray,
    layout: RegionLayout,
    *,
    target_bytes: int,
    max_source_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
```

It returns:

- `batch_region_edges[B + 1]`;
- `batch_piece_edges[B + 1]`; and
- structured `pieces[P]` with `source_block`, `source_start`, `source_stop`.

All plan arrays share one safe signed dtype: int32 when every bound fits, otherwise
int64. Packed/output destinations and non-position chunk shapes are derived rather
than stored. The planner ends with vectorized postcondition checks; there is no
second validation interface.

Private defaults are `_TARGET_BYTES = 128 * 1024**2` and
`_MAX_SOURCE_BLOCKS = 16`. Private helpers accept overrides for tests. Dataset and
DataTree accessors expose no tuning arguments initially.

For Dask input, inspect public `values.chunks` and derive cumulative position-block
boundaries. For `chunks=None`, inspection does not access `values.data`; the eager
gather path plans from shape and slices backend values only when executing the
selected pieces.

One piece represents one region/source-position-block intersection. Batches split
only at complete-region boundaries and honor both private limits. One individual
region may exceed a limit when it alone is larger or spans too many source blocks;
that region stays whole in one oversized batch. Sharing a source block never allows
multiple otherwise splittable regions to exceed a limit.

Planner invariants are:

```text
batch region and piece edges are monotone and complete
every piece has a valid block and positive in-bounds source slice
packed rows are covered exactly once per non-position block
output position boundaries are packed region offsets
non-position Dask chunks equal input chunks
```

Planning memory is `O(n_regions + n_pieces + n_batches)` in typed arrays. CI
constructs the complete immutable layout and all three plan arrays for two million
one-base regions against logical two-trillion-position scalar and 76-column sources
without allocating source values. It asserts signed typed storage plus exact region,
piece, and batch counts. Graph task and pickle growth are tested with 512 and 1,024
regions, bounded by source blocks and batches; serialized tasks contain neither a
source path nor an open callable. The test never serializes a two-million-region
graph or pins a machine-specific byte ceiling.

## Gathering

One pure NumPy kernel receives already-computed blocks, a bounded structured-piece
view, and output shape. It derives output destinations in piece order, allocates one
block, and copies slices. It has no path, Zarr, xarray, scheduler, or environment
dependency.

For Dask, the assembler calls `values.data.to_delayed()` only after
`values.chunks is not None`. It retains each delayed source-block object and passes
the exact same object/key to every dependent output task. Each output
position-batch × non-position-block task depends only on the blocks it needs.
Construction reads no values. A scheduler may recompute an evicted dependency;
PBZ guarantees one shared graph key, not decode-once execution.

For `chunks=None`, bulk `regions()` is deliberately eager and reads bounded source
pieces. It slices each selected backend piece before NumPy conversion, never
coerces the full source, and returns ordinary eager selected values. The complete
result still occupies memory proportional to all selected bases × columns ×
variables and must fit RAM; use default Dask for large selections. Unlike the Dask
path, it does not promise zero reads during `regions()` construction. This is
distinct from single `region()`, which remains xarray backend-lazy.

## Packed validation and reduction

One private validator checks the packed marker, unindexed geometry, strict offsets,
positive interval lengths, complete provenance, storage index, and position-first
variables. It reads no data variables, marks the validated offsets read-only, and
returns only that ndarray. There is no packed-layout object.

Reduction aligns work to complete-region groups. Inside each bounded task it creates
a local xarray DataArray and segment labels, obtains an xarray groupby object, and
calls the named method from the allowlist with `dim="position"` and user kwargs.
Xarray may use flox internally. PBZ does not call flox functions directly or define
a reducer registry.

For each variable, position becomes region. Region provenance/storage index remain;
packed offsets and representation attrs are removed. Xarray owns result dtype, NaN,
`skipna`, `ddof`, quantile dimensions, attrs, and derived encoding behavior. PBZ
does not copy `_FillValue` conditionally.

Non-position operations are ordinary xarray operations and may run on either side
of PBZ reduction:

```python
position_then_column = packed.pbz.reduce("mean")["signal"].max("context")
column_then_position = packed.max("context").pbz.reduce("mean")["signal"]
```

The order is meaningful for nonlinear composition: the first expression takes the
largest column mean in each region, while the second averages the per-position
column maxima. Keep the complete packed Dataset through `pbz.reduce`; selecting a
single packed data variable can make xarray drop the disconnected region
provenance that validation requires. Select one variable freely from the ordinary
reduced result.

An unindexed position axis cannot expose every length-preserving reversal or
permutation. Such transformations invalidate PBZ semantics even if validation
cannot detect them. Structurally detectable slicing/sorting is rejected.

## Lifetime and output size

Derived Dask values borrow the lifetime of their source Dataset/DataTree. PBZ
preserves xarray close callbacks and adds no lease or reopen path. Compute or persist
while the owner remains open.

Two million regions × six variables × 76 columns is 912 million values. A float64
summary is about 6.80 GiB before xarray/Dask overhead, so documentation must not
suggest unconditional full `.compute()`:

```python
summary.isel(region=slice(0, 10_000)).compute()
summary[["coverage"]].compute()
summary.to_zarr("summary.zarr")
```

Dask enables parallel reduction; an output table format does not. Ordinary xarray
output is not promoted to a PBZ region store.

## Test policy

Retain tests for public xarray/write behavior, storage/resource boundaries,
reproduced regressions, and load-bearing planner invariants. Avoid reducer/backend
cross-products, private class-shape tests, display setup, scheduler/logging behavior,
or full-scale graph serialization. Durable coverage is organized by behavior, not
a mandated final filename set.

## Acceptance criteria

1. Single regions slice existing values strictly and lazily.
2. Many regions normalize to one read-only `RegionLayout` and three typed plan
   arrays per variable; no other planning class exists.
3. Dask gathering is lazy and touched-block-only, reuses one delayed source key
   across dependents, honors private batch limits, and contains no reopen path.
4. `chunks=None` gathering reads selected pieces only but its complete eager output
   must fit RAM.
5. Packed position chunks are region-complete and non-position Dask chunks survive.
6. Named xarray reductions match representative mean/std/median/quantile behavior.
7. Reduced outputs are ordinary safely sortable xarray values.
8. Large planning and smaller structural graph bounds are both covered.
