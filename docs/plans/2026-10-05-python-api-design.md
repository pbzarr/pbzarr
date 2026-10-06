# Python API design for format 0.5

Status: agreed design, 2026-10-05. Not yet implemented. This replaces the
earlier `Dataset`/`Track` wrapper draft in
[proposals/format-05/PYTHON_API.md](../proposals/format-05/PYTHON_API.md),
which was written before format 0.5 landed and before the genomic index was
settled. The [schema](../proposals/format-05/README.md) owns the on-disk
contract. The fuller store specification lives in `docs/STORE_SPEC.md`, which is
untracked and present only in the main checkout.

## Why this replaces the earlier draft

Two facts settled since that draft was written change the shape of the API.

A format 0.5 store opens as one ordinary Xarray Dataset. Tracks are root data
variables over shared root coordinates, so there is no per-track group to
reassemble and no composition step.

A genomic index carries the reference. `GenomicIndex` subclasses the public
`xarray.Index`, stores rows as a `pandas.RangeIndex` (136 bytes for a
3.1-billion-position genome), and survives `isel`. The original argument for a
`Track` wrapper was that an extracted `DataArray` loses its reference arrays.
The index removes that argument, because the reference travels with the array.

`IndexSelResult` is exported from `xarray` at top level, so the index needs no
private Xarray import. The prototype's `from xarray.core.indexing import
IndexSelResult` is simply the wrong path.

## The invariant

`pbz.Dataset` is a store handle, not a value. Values are ordinary Xarray
objects.

The handle owns lifetime, index attachment, the store catalog, and the
operations that need store state or genomic geometry. It has no arithmetic, no
comparisons, no reductions, no `astype`, no `with_attrs`, no `item`, and no
method that takes a value and returns a wrapped value. Those live on the Xarray
objects the handle gives out.

This is deliberate. The abandoned format-0.4 wrapper carried about thirty
one-line delegators, and a review found they had already drifted into five
different attribute and encoding policies. The remedy is to not have them.

## Public surface

```python
def open(source, *, mode="r", tracks=None, chunks=..., storage_options=None) -> Dataset: ...
def create(destination, *, reference, coords=None, attrs=None,
           storage_options=None) -> Dataset: ...
def drop_index(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray: ...


class Dataset:
    reference: Reference
    tracks: Mapping[str, xr.DataArray]
    coords: Mapping[str, xr.DataArray]
    sizes: Mapping[str, int]
    attrs: Mapping[str, object]

    def __getitem__(self, name: str) -> xr.DataArray: ...
    def to_xarray(self) -> xr.Dataset: ...
    def close(self) -> None: ...
    def __enter__(self) -> Dataset: ...
    def __exit__(self, *exc) -> None: ...

    def windows(self, size: int) -> Windows: ...
    def write(self, tracks: Mapping[str, xr.DataArray], *, encoding=None) -> None: ...
    def reduce_regions(self, intervals, reducer, /, **kwargs) -> xr.Dataset: ...
    def regions(self, intervals) -> xr.Dataset: ...
```

`ds["depth"]` returns a plain `xr.DataArray` with the index attached. Selection
is native: `ds["depth"].sel(position="chr1:100-1000", sample=["s1"])`. There is
no `Dataset.select`, so there is no second selection path to keep in agreement
with Xarray's.

`mode="r"` is read-only. `mode="r+"` opens an existing store for `write`; it
does not create a missing destination. `create` fails if its path exists.

`tracks=` narrows which measurements the handle exposes. It does not narrow the
catalog used for write validation.

## Index lifecycle

### Attach

`open` attaches a `GenomicIndex` to every genomic dimension present, built from
`pbz:reference`. Two conditions are store-validity checks and must raise with
that framing:

1. The dimension must span the reference. `position` is `sum(L_i)` and `bin_N`
   is `sum(ceil(L_i / N))`. A mismatch means the store is invalid.
2. An existing `position` or `bin_N` coordinate is refused. Format 0.5 requires
   stores to omit coordinate variables for genomic dimensions, so finding one
   means the store is invalid.

### Ride

The index travels through Xarray without help:

| Operation | Index |
| --- | --- |
| `sel(position="chr1:100-200")` | returns a slice, index narrows |
| `isel(position=slice(...))` | rows narrow |
| `isel(position=3)`, 0-d indexer | removed, the axis is gone |
| reduction over the genomic axis | removed, the axis is gone |
| arithmetic between two indexed arrays | `join` refuses unequal indexes |

That last row is an alignment guard worth more than it looks: combining a chr1
view with a chr2 view raises instead of producing values that look plausible.

Some Xarray operations drop indexes. After that the row mapping cannot be
recovered from the array alone, because a partial view's rows are not derivable
from its length. The rule is therefore: **fail loudly at the point of use,
never silently reconstruct.** `windows`, `write`, `reduce_regions` and
`regions` each resolve the index and raise a message that names the array and
says the genomic context was dropped.

### Drop

Dropping is internal and happens in exactly two places:

1. Before `to_zarr`. A custom index does not serialize, and format 0.5 forbids
   a stored genomic coordinate, so the drop satisfies the specification rather
   than working around a limitation.
2. Before building the packed-regions form, whose contract requires `position`
   and `region` to be unindexed.

`to_xarray()` keeps the index, because carrying genomic context into native
work is the reason the index exists. A caller who then runs `to_zarr`
themselves will hit an Xarray error, so `pbz.drop_index(obj)` is public for
that case.

## Contig-aware windowing

`ds.windows(size)` returns a small object exposing `mean`, `sum`, `min` and
`max`, each returning an `xr.Dataset` on a `bin_<size>` dimension.

Plain `coarsen` is wrong here: format 0.5 bins restart at zero in every contig,
so a window must not span a contig boundary. For local bin `k`,
`start = k * size` and `end = min((k + 1) * size, contig_length)`, which means
the last bin of a contig can be short.

`windows` requires a complete track, matching the Rust rule. Its result
therefore always spans the reference in `bin_<size>` units and carries its own
`GenomicIndex`, which makes the result a writable format 0.5 track.

## Writes

`ds.write({"callable_count": result})` takes plain `xr.DataArray` values on an
`r+` handle. Validation, which is the specification read back as checks:

- The name must not collide with any variable, coordinate or dimension in the
  full catalog, including variables a `tracks=` filter hid.
- A genomic dimension must be first and be `position` or `bin_N`, and must span
  the reference. Non-genomic results and scalars have no first-axis rule.
- Shared non-genomic dimensions must have identical labels in the same order.
  A new dimension needs a coordinate array of the same name with unique,
  nonmissing labels.
- The dtype must be supported. `_FillValue` is written only where the caller
  declared it.
- Attributes must be JSON-serializable and must not use the `pbz:` namespace or
  the names `zarr_conventions` or `coordinates`.

The write then drops the index and uses `to_zarr(mode="a")`. No Rust is
involved. Writing 0.5 stores from Xarray is already proven in both directions:
the prototype writes stores the Rust checker validates, and the Rust writer
produces stores native Xarray reads.

Errors may leave the store without the new variable. The write is not
transactional and must say so.

## Region reduction and packed reads

`reduce_regions` and `regions` reuse `python/pbzarr/_regions.py`. Its planner is
arithmetic over `(contigs, offsets)` and is format-neutral. Only two pieces are
format-coupled:

1. `_validate_regions_dataset` requires the Dataset's attributes to be a subset
   of `{genome_checksum, coordinates, genome_name}`, so a 0.5 root is rejected.
   A new entry accepts `zarr_conventions`, `pbz:version`, `pbz:reference` and
   user attributes, and derives `(contigs, offsets)` from the reference by
   prefix sum instead of reading `contigs` and `offsets` coordinate arrays.
2. The packed form requires `position` unindexed, so the index is dropped
   before packing.

`_resolve_regions`, the gather path and `_reduce_packed_dataset` are unchanged.

The result is a plain `xr.Dataset`, one row per region, carrying
`region_contig`, `region_start` and `region_stop`. It has no genomic index,
because `region` is not a genomic dimension.

The chunk-alignment check in `_validate_packed_dataset` applies to the packed
representation, whose `offsets` are region boundaries. It is an internal
invariant of a layout pbzarr constructs and is not affected by the store
format.

## Module layout

Split by format rules versus value types, not by read versus write. The 0.4
implementation split by direction and ended with format rules stranded in the
write module, which produced six separate duplication findings.

| Module | Owns |
| --- | --- |
| `python/pbzarr/_index.py` | `GenomicIndex`, `Reference`, genomic dimension names and bin arithmetic |
| `python/pbzarr/_dataset.py` | the `Dataset` handle, `open`, `create`, `drop_index` |
| `python/pbzarr/_windows.py` | contig-aware windowing |
| `python/pbzarr/_write.py` | extended with derived-track writes |
| `python/pbzarr/_regions.py` | extended with the 0.5 validation entry |

Region parsing reuses `_regions.py`'s `_resolve_one`. The prototype's
`Reference.region_slice` duplicates it, and the equivalent duplication in the
0.4 attempt was a review finding.

`Reference` holds names and lengths, validates them in one pass into locals, and
assigns each field once. The 0.4 version rewrote a frozen dataclass four times
and fused six rules into two messages, which four reviewers independently
flagged. `offsets` and `size` are `cached_property`, because the 0.4 version
recomputed them on every access inside the region hot path.

## Out of scope

Deferred in the roadmap or undefined in the schema, and therefore not built
here: `replace`, `patch`, `append`, `update_attrs`, multiscale and pyramid
support, conversion from format 0.4, and 64-bit integer dense dtypes.

`Dataset.from_xarray` is dropped: it would mean a store handle with no store.

Registering the index for native `ds.sel` use by callers who never import the
wrapper is the behavior this design already provides, since the index is a real
Xarray index. No separate opt-in accessor is needed. Note that this does
conflict with `ROADMAP.md`, which records that no custom index is approved;
that entry is now out of date and should be updated when this lands.

## Testing

Core functionality only, no edge-case or attribute tests. About six:

1. `open` attaches the index, and `sel(position="chr1:100-200")` returns the
   expected rows and values.
2. `windows(size).mean()` across a contig boundary, where the last bin of the
   first contig is short. This is what plain `coarsen` gets wrong.
3. A derived write round-trips: compute `(depth >= 5).sum("sample")`, write it,
   reopen, compare.
4. `reduce_regions` over a few intervals matches a hand-computed value.
5. An array whose index was dropped raises a clear error from `windows` and
   from `write`.
6. Arithmetic between two differently-selected views raises rather than
   aligning.

Plus one cross-language check: write a track from Python, then validate the
store with the existing `check_dataset` Rust example. That is the strongest
available test and costs almost nothing, because the example already exists.

## Details to settle during implementation

- Whether `write` refreshes consolidated metadata. It must leave the store in
  the same state the Rust writer leaves it in, so that both paths agree.
- The exact `chunks=` default for `open`. The format-0.4 path uses a sentinel
  meaning "use the stored chunking"; `xr.open_zarr` spells that `chunks={}`.
- Whether `Windows` exposes `reduce(func)` in addition to the four named
  reducers. The prototype has it; it is not needed for the tests above.
