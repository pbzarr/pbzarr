# Python API proposal for the PBZ core draft

Status: discussion draft, 2026-09-26. This describes a proposed API for the
[core convention draft](README.md), not the implemented format 0.4 API.
Method names and signatures below are proposals. No production behavior or
release scope changes with this document.

Follow-up: the [runnable Dataset/Track prototype](../../../python/prototypes/dataset_track_poc/README.md)
explores this interface using a custom `GenomicIndex` in place of the public
`geometry` argument below. It demonstrates wrapper/native calculations,
contig-aware coarsening, and new-track writes. It is an experiment, not adoption
of the draft API or a change to production support.

Decisions recorded 2026-09-30 supersede the earlier signatures below: use a
retained runtime genomic index instead of the public `geometry`/`GenomicMapping`
surface, and let the Track wrapper own contig-aware windowing (working name
`windows`). A separate native coarsening accessor is optional. The unchanged
signatures below remain the earlier API sketch, not the final class definitions.
The [conformance draft](CONFORMANCE.md) now owns concrete store examples, and
[Rust/CLI decisions](RUST_CLI.md) records the agreed cross-language scope.

## Recommendation

Use two main objects, composed around ordinary Xarray objects:

* `Dataset` represents the reference, coordinates, and named tracks together.
* `Track` represents one measurement or computed result, including a scalar.

PBZ owns genomic selection, retained reference context, compatibility checks,
and persistence. Xarray and its underlying arrays own numerical computation.
The same genomic helpers back registered `.pbz` accessors for native users.
`to_xarray()` is the explicit route to the rest of Xarray's API.

The introductory documentation should consistently teach the wrappers. An
interoperability chapter teaches the native accessors and conversion boundary.

## The ordinary workflow

```python
import pbzarr as pbz

with pbz.open("cohort.pbz", mode="r+", chunks="auto") as ds:
    depth = ds["depth"]
    callable_count = (depth >= 5).sum(dim="sample")
    callable_count = callable_count.with_attrs(
        description="Number of samples with depth at least five.",
        notes="Computed across all samples currently in this dataset.",
    )

    ds.write({"callable_count_ge5": callable_count})
```

Opening and constructing this expression leave measurement data lazy when
using Dask. `write` executes the calculation and writes its output without
requiring a preceding whole-result `compute()`. The call returns when writing
finishes. It creates a root data variable named `callable_count_ge5`.

`sum` forwards to `DataArray.sum`; PBZ handles result context and validation.
There is no PBZ summation kernel, expression language, or provenance graph.
Notes are ordinary user attributes and are stored with the new variable.

## Public shape

The signatures illustrate the intended boundary rather than every overload.
`ArrayLike` includes supported NumPy, Dask, and other duck arrays. Reference
inputs are ordered mappings of contig names to lengths, or the immutable
reference object exposed by another Dataset.

```python
def open(
    source, *, mode="r", tracks=None, chunks="auto",
    array_backend="numpy", storage_options=None,
) -> Dataset: ...

def create(
    destination, *, reference, coords=None, attrs=None,
    storage_options=None,
) -> Dataset: ...


class Dataset:
    reference: Reference
    tracks: Mapping[str, Track]
    coords: Mapping[str, xr.DataArray]
    sizes: Mapping[str, int]
    attrs: Mapping[str, JSONValue]

    def __getitem__(self, name: str) -> Track: ...
    def select(self, indexers=None, *, region=None, **labels) -> Dataset: ...
    def to_xarray(self) -> xr.Dataset: ...
    def compute(self, **kwargs) -> Dataset: ...
    def close(self) -> None: ...

    @classmethod
    def from_xarray(cls, dataset: xr.Dataset) -> Dataset: ...

    # Store mutations; available on an unselected writable root handle.
    def write(self, tracks: Mapping[str, Track], *, encoding=None) -> None: ...
    def update_attrs(self, attrs, *, track=None) -> None: ...

    # Later mutation support, with deliberately separate meanings.
    def replace(self, tracks: Mapping[str, Track], *, encoding=None) -> None: ...
    def patch(self, tracks: Mapping[str, Track]) -> None: ...
    def append(self, *, dim: str, labels, tracks, coords=None) -> None: ...


class Track:
    def __init__(
        self, data: ArrayLike, *, dims, coords=None,
        reference=None, attrs=None,
    ): ...

    name: str | None
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    sizes: Mapping[str, int]
    dtype: DType
    coords: Mapping[str, xr.DataArray]
    attrs: Mapping[str, JSONValue]
    reference: Reference | None
    geometry: GenomicMapping | None

    def select(self, indexers=None, *, region=None, **labels) -> Track: ...
    # Explicit comparison, arithmetic, and Boolean operator adapters.
    def sum(self, dim=None, **kwargs) -> Track: ...
    def mean(self, dim=None, **kwargs) -> Track: ...
    def min(self, dim=None, **kwargs) -> Track: ...
    def max(self, dim=None, **kwargs) -> Track: ...
    def any(self, dim=None, **kwargs) -> Track: ...
    def all(self, dim=None, **kwargs) -> Track: ...
    def astype(self, dtype, **kwargs) -> Track: ...
    def with_attrs(self, **attrs) -> Track: ...
    def compute(self, **kwargs) -> Track: ...
    def item(self): ...
    def to_xarray(self) -> xr.DataArray: ...

    @classmethod
    def from_xarray(
        cls, array: xr.DataArray, *, reference=None, geometry=None,
    ) -> Track: ...
```

The wrappers do not subclass Xarray or forward arbitrary unknown attributes.
Each supported adapter has an explicit return type and geometry rule.
Additional native operations remain accessible through `to_xarray()`.
This follows Xarray's preference for
[composition and accessors](https://docs.xarray.dev/en/stable/internals/extending-xarray.html)
without attempting to reproduce its entire interface.

`Reference` and `GenomicMapping` are small immutable context values, not new
storage objects. Reference contains names and lengths. Mapping additionally
identifies the genomic dimension and the selected rows in their current order.
Mappings should use compact slices/runs rather than whole-genome coordinate
arrays. They are runtime state and introduce no required Zarr attributes.
Wrappers also retain private selection mappings for labeled non-genomic axes,
so writers can distinguish a known sample selection from an unrelated smaller
array. Native conversion does not automatically transfer those mappings back.

## Opening, inspecting, and selecting

`open` returns one predictable type. An optional `tracks=` filter limits opened
measurements while retaining the reference and needed coordinates. Default
mode `"r"` is read-only. Mode `"r+"` opens an existing store for explicit changes;
it does not create a missing destination. `create` fails if its path exists.
Even when `tracks=` limits exposed measurements, a writable handle validates
against the complete store catalog. Name collisions and shared coordinates
include hidden variables, and future append extends hidden affected arrays too.

The Dataset representation lists tracks with their dimensions, sizes, dtype,
and lazy/eager state. It also displays the reference and current selection.
`ds["depth"]` and `ds.tracks["depth"]` return equivalent Track objects;
coordinates live in `ds.coords` and do not appear as tracks.

```python
with pbz.open("cohort.pbz") as ds:
    view = ds.select(
        region="chr1:100-1000",
        sample=["sample_1", "sample_3"],
    )
    depth = view["depth"]
    same_depth = ds["depth"].select(
        region="chr1:100-1000",
        sample=["sample_1", "sample_3"],
    )
```

All region forms use zero-based, half-open bounds, including strings. A tuple
`("chr1", 100, 1000)` is an equivalent explicit form. Unknown contigs, invalid
bounds, missing labels, duplicate requested labels, and unknown dimensions raise
errors. Generic label selection excludes the reference dimension; use `region=`
to select genomic data without shortening the reference. The optional `indexers`
mapping selects ordinary labeled axes and supports names that conflict with API
keywords, for example `select({"region": ["region_1"]})` on a non-genomic result.
Specifying the same axis in both forms is an error.
A scalar label selection preserves a length-one dimension; reduction
is the explicit way to remove an axis in the initial wrapper API.

Selection returns a new view. A Dataset applies each selector to the variables
carrying that dimension; variables without it remain present and unchanged.
For example, a sample selector does not alter an existing sample-reduced count.
A region selector slices every genomic dimension using its own geometry.
On binned tracks, it selects complete bins overlapping the region; it does not
recalculate values for clipped bin edges. Repeated selection intersects the
current view and cannot expand it.

Initially support a single contiguous region per selection. Disjoint and
reordered genomic selections require an additional mapping contract; they need
not complicate this first interface. Label selections retain requested order.

The reference remains the full reference even in a regional view. The view's
mapping records which bases or bins it contains. Its displayed shape is the
selected shape, not a newly defined reference length.

Keep lazy computation and writing inside the owning `with` block. Closing the
owner closes its store resources; a view does not silently reopen them.
`compute()` returns a detached, materialized wrapper retaining its context.

## Computation and attributes

Operators support scalars, compatible Tracks, and non-genomic labeled
DataArrays such as a threshold coordinate. Shared dimension labels must match
exactly, including order. Shared genomic axes must have the same reference and
row mapping. Different genomic dimensions require an explicit conversion;
`position` and `bin_1000` must not accidentally broadcast into two genomic axes.
Unlabeled nonscalar operands should first be given named dimensions.

PBZ prevents implicit inner joins or coordinate unions in these adapters. Users
wanting intentional realignment can perform it explicitly in native Xarray and
supply the resulting genomic context on return.

```python
with pbz.open("cohort.pbz", mode="r+") as ds:
    # The example schema already contains threshold(threshold) = [5, 10].
    callable_mask = ds["depth"] >= ds.coords["threshold"]
    counts = callable_mask.sum(dim="sample")
    means = ds["depth"].mean(dim="position")

    ds.write({
        "depth_at_threshold": callable_mask,    # position, sample, threshold
        "sample_count_at_threshold": counts,   # position, threshold
        "mean_depth_per_sample": means,         # sample
    })
```

These destination names must be unused. Ordinary named broadcasting supplies
the new threshold axis. PBZ normalizes genomic dimensions to first position;
otherwise native broadcasting may choose a different dimension order.

Reductions use Xarray's `dim`, missing-value rules, and supported keyword
arguments, including backend-specific arguments where available. They always
return a Track. Removing the genomic dimension removes its row mapping, while
retaining reference context; removing all dimensions yields a scalar Track.
For example, `counts.sum().compute().item()` obtains a Python scalar explicitly.
`item()` requires a materialized scalar and never triggers hidden computation.

Selection preserves attributes. Computations should start derived results with
empty user attributes, so a description or units field is not incorrectly
inherited by a different measurement. Users attach the intended metadata with
`with_attrs`, which merges attributes into a new Track. Encoding details such
as source dtype and missing sentinel are not blindly copied after computation.

Dataset and Track `attrs` are read-only mappings in the wrapper API.
`ds.update_attrs({...})` explicitly changes root user attributes on disk;
`ds.update_attrs({...}, track="depth")` changes an existing track's attributes.
These merge user metadata and reject changes to convention fields, dimension
definitions, and encoding fields. Root coordinates have no special grouping
semantics. Grouping remains ordinary Xarray usage, covered in tutorials.

## Creating and writing

`create` establishes a root Dataset with its reference and any supplied dimension
coordinates. It returns a writable Dataset. A new dimension can also be
introduced by the first Track that uses it, provided that Track supplies labels.

```python
with pbz.open("cohort.pbz") as source:
    selected = source["depth"].select(sample=["sample_1", "sample_3"])

    with pbz.create("subset.pbz", reference=source.reference) as output:
        output.write({"depth": selected})
```

The new Dataset has two samples and the full original reference. A selected
genomic region would still be placed into the full reference with fill outside
the selection. A genuinely changed reference requires an explicit new reference
and explicit remapping; selection does not rename or shorten contigs.

`write` accepts a mapping to support several related results in one call. It
checks the whole request before mutation, then creates the requested variables.
It validates reference order/lengths, named dimensions, exact shared labels,
coordinate classification, supported encoding, and known placement. It must not
silently merge different labels or overwrite an existing variable. Validating
the batch does not make it an atomic transaction.

Writing happens through the unselected root Dataset. Views and materialized
results are analysis objects and do not carry implicit write targets. A
successful mutation refreshes the writer's catalog; callers obtain fresh Tracks
from it for subsequent work. Existing native objects are not live subscriptions.

Per-variable `encoding` uses supported native Xarray/Zarr encoding keys, such
as `dtype`, `chunks`, `fill_value`, and optional `_FillValue`. Execution chunking
and storage chunking remain separate choices. Dtype conversion must be explicit
and checked where loss is possible. Reductions are not silently downcast to a
source dtype.

For example, a regional count can be stored under a new name:

```python
with pbz.open("cohort.pbz", mode="r+") as ds:
    depth = ds["depth"].select(region="chr1:100-1000")
    count = (depth >= 5).sum(dim="sample").with_attrs(
        notes="Only chr1:[100,1000) was evaluated. Other positions store zero.",
    )
    ds.write(
        {"regional_count": count},
        encoding={"regional_count": {"fill_value": 0}},
    )
```

The destination has the full `position` length. The writer places these values
at their original rows and warns that zero elsewhere means the declared fill,
not a computed count. Require an explicit fill choice for partial new-track
writes. An optional missing sentinel is a separate decision; storage fill alone
does not make zero missing. The writer allocates metadata and schedules touched
chunks without constructing a full dense fill array. Partial chunk writes need
safe coordination even when all-fill chunks can be omitted.

Selected columns can similarly be placed into an existing dimension's label
positions. This uses an explicit mapping and fills other columns. A complete
array with a reordered or changed shared coordinate set is rejected, rather
than treated as an inferred sparse write. New dimensions use supplied labels.

`replace` and `patch` are separate later capabilities. Replacement changes the
entire named variable; patch changes only specified existing cells and preserves
the rest. Neither spelling is an alias for creating a new track. Replacement
or patching must not modify arrays still needed by a lazy source calculation:
implementation must stage such results safely or reject that case before mutation. These
capabilities need not block initial new-track writing.

## Independent binned tracks

Writing a binned Track does not require a base-resolution source:

```python
with pbz.open("cohort.pbz", mode="r+") as ds:
    # values is an externally computed array, here with shape (3, 3).
    signal = pbz.Track(
        values,
        dims=("bin_1500", "sample"),
        coords={"sample": ["sample_1", "sample_2", "sample_3"]},
        reference=ds.reference,
        attrs={"description": "Signal measured in 1500-base contig-local bins."},
    )
    ds.write({"external_signal": signal})
```

For the draft reference lengths 1500 and 2200, there are three such bins.
Construction declares canonical full-reference bin order and validates extent.
Any positive width can be used; there is no required relation to other tracks.

A future convenience helper could be `depth.bin(width=1000, reduction="mean")`.
It would own contig segmentation and delegate each numerical reduction. It must
respect contig-local origin and final clipped bins, including when the input is
a regional view. Initially requiring full-reference input avoids pretending a
partially observed bin is complete. This helper is optional; construction,
selection, and writing of independent binned tracks are core workflows.
Native `.coarsen()` over the concatenated array is not a genomic binning promise.

## The native Xarray boundary

Export is lazy and preserves supported underlying array types. It creates an
independent Xarray container/metadata view; copying all measurement data is not
required. Users must not mutate shared backing buffers in place. A full Dataset
export carries root reference coordinates. A Track export usually cannot carry
those unrelated coordinate dimensions.

```python
with pbz.open("cohort.pbz", mode="r+") as ds:
    depth = ds["depth"].select(region="chr1:100-1000")
    native = depth.to_xarray()
    result = native.quantile([0.25, 0.5, 0.75], dim="sample")

    quantiles = pbz.Track.from_xarray(result, geometry=depth.geometry)
    ds.write(
        {"depth_quantiles": quantiles},
        encoding={"depth_quantiles": {"fill_value": float("nan")}},
    )
```

Passing `geometry=` explicitly asserts that the surviving genomic rows retain
that mapping and order. PBZ checks dimensions, extents, and labeled coordinates,
but cannot prove this assertion for arbitrary native computations. Equal-length
reversal or permutation is not detectable from shape. If native code changes
genomic rows, callers must supply a correspondingly changed mapping or use PBZ
selection before escaping. Initially unsupported mappings raise an error.

For an externally supplied full-reference DataArray, pass `reference=` instead;
this explicitly declares canonical full-reference order. Genomic input lacking
either declaration is rejected. A non-genomic DataArray needs neither.
This initial native constructor does not restore prior column-selection state.
Writing it against existing labeled dimensions requires exact destination
labels; partial-column scattering uses context retained by wrapper operations.

`Dataset.from_xarray(native)` validates an existing full Dataset against the
core convention and returns an in-memory wrapper. It requires its reference
arrays, convention attributes, and complete genomic extents. It does not infer
a store write target. Individual results can then be written through a
destination created or opened with PBZ.

`Dataset.to_xarray()` on a regional view returns an analysis Dataset with a
shortened genomic axis, not a directly serializable core PBZ store. It must not
claim to be a valid full-reference PBZ encoding: remove the PBZ convention/version
markers on such exports, and retain placement only in the original wrapper.
Native attributes are never proof of row identity. Write through the PBZ writer
to create the full-reference destination required by the core draft.

The Dataset accessor provides native equivalents of genomic helpers:

```python
native = xr.open_zarr("cohort.pbz", chunks="auto")
selected = native.pbz.select(region="chr1:100-1000")
reference = native.pbz.reference
```

For a detached DataArray, require explicit context, for example
`array.pbz.select(region=..., geometry=track.geometry)`. A full native Dataset
can establish reference context from its root arrays. Selected native objects
need an explicit mapping for subsequent genomic operations; accessor caches
must not pretend to survive arbitrary Xarray operations. Xarray documents that
[new objects receive new accessor instances](https://docs.xarray.dev/en/stable/internals/extending-xarray.html).
Both surfaces call the same selection/validation implementation. Custom indexes
can improve this later without becoming prerequisites or changing the schema.

## Dask and array backends

Keep three choices separate:

* `chunks` selects lazy partitioning. `"auto"` or a dimension mapping uses Dask;
  `None` disables Dask and must not be advertised as an eager whole-store load.
* `array_backend` selects the arrays used for measurement chunks. Start with
  `"numpy"`; an installed future plugin could provide `"cupy"`.
* The Dask scheduler or distributed client is configured through Dask itself.

Thus a future GPU workflow keeps the analysis unchanged:

```python
# Illustrative future plugin support, not an available backend today.
with pbz.open("cohort.pbz", chunks="auto", array_backend="cupy") as ds:
    count = (ds["depth"] >= 5).sum(dim="sample")
```

The backend adapter owns supported chunk conversion, device transfer, and
per-chunk serialization transfer. The wrapper must preserve duck arrays and
avoid measurement-wide `.values`, `np.asarray`, or implicit CPU conversion.
Reference and dimension labels may remain host metadata. NumPy remains a
dependency; measurement kernels need not be NumPy-only.

Xarray already
[delegates array methods to wrapped array implementations](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html),
but some operations fall back to NumPy. Therefore GPU support needs verified
operation/encoding capabilities and explicit unsupported-operation errors; it
cannot be promised for every native Xarray method. Xarray's
[chunk-manager arguments are experimental](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html),
so keep them behind a small internal adapter rather than exposing that protocol
as PBZ's permanent public plugin contract.

Native [Xarray/Zarr writing](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html)
supplies the foundation for lazy writes. PBZ must additionally plan coordinate
metadata and safe region/chunk placement. Rechunking, reductions, and backend
transfers affect memory; avoiding whole-result materialization is a requirement,
not a fixed peak-memory guarantee. A separate sink framework is not part of
this proposal.

## Future column append

Reserve one dimension-explicit operation with the agreed simple behavior:

```python
with pbz.open("cohort.pbz", mode="r+") as ds:
    ds.append(
        dim="sample",
        labels=["sample_4", "sample_5"],
        tracks={"depth": new_depth},
    )
```

`new_depth` is a Track with those two labels in that order, the same reference,
and unchanged other dimensions. Initially require full genomic coverage for
append inputs; regional new-track writes remain a separate operation.

Append validates the complete request, adds unique nonmissing labels, and extends
every root array using `sample`. Supplied values populate their arrays; omitted
arrays and auxiliary coordinates use their declared fill. `coords=` optionally
supplies values for affected auxiliary coordinates. The dimension coordinate
receives the new labels. Arrays without `sample` remain unchanged, including
previous callable counts: they do not acquire an automatic recomputation rule.

The same operation works for other labeled non-genomic dimensions, such as
`threshold`. It cannot append contigs or extend `position`/`bin_N`. Generic
`column_dim` metadata is unnecessary. Importers may retain `column_dim=` to name
their output axis; multidimensional analysis always names its actual dimensions.

Append changes shapes. Existing lazy Tracks/views must be invalidated or kept
under an explicit safe lifetime policy before releasing this operation. Prefer
rejecting use of stale wrappers after a structural mutation and obtaining fresh
Tracks from the writer. Detached native arrays cannot receive that protection;
document that they must be reopened. A writer must finish source reads before
resizing arrays those reads depend on, or reject that dependency.

Initial mutation support assumes one writer and does not promise rollback,
multi-array atomicity, or crash recovery. Omitted data uses existing fill values;
if the required fill/encoding cannot represent the append, fail before mutation.
No additional provenance or transaction schema is introduced.

## Alternatives considered

An accessor-first design has the smallest custom surface:
`xr.open_zarr(...).pbz.select(...)`, ordinary native computation, and a PBZ write
helper. It exposes Xarray immediately and makes context after native operations
the caller's responsibility. Keep it as the interoperability surface.

A separate Store design makes storage authority especially clear:
`store = pbz.Store(path); ds = store.read(); store.write({"count": result})`.
It cleanly separates storage lifetime from analysis, but adds a third main
object to every introductory workflow. Borrow its explicit write boundary by
restricting mutations to the unselected writable Dataset handle.

The recommended wrapper design keeps `ds["depth"]`, comparisons, named
reductions, and `ds.write(...)` together. It hides genomic placement and storage
validation while retaining one native array model. Its cost is maintaining a
small, explicit set of adapters and honest limits at the Xarray boundary.

## Scope and validation

The first useful slice is open/create, track access, single-region/label
selection, the callable-count operations, manual metadata, native conversion,
and new-track writes. It includes independent binned inputs and lazy Dask
execution. GPU adapters, append, replacement, patching, and genomic indexes can
follow without changing the core metadata schema. This ordering is a proposal
within this design, not an edit to the current release roadmap.

The [existing schema probe](VALIDATION.md) checks the underlying native layout,
lazy calculations, multidimensional counts, independent bins, and a small
fill-backed regional write. It does not test these unimplemented wrapper methods,
GPU execution, or append. Proposal examples should be syntax-checked and local
documentation links checked; full wrapper acceptance tests belong to a later
implementation task. For this draft, all 11 Python code blocks parsed and the
six local link occurrences across this document and the core proposal resolved.
These are documentation checks, not evidence that the proposed API exists.
