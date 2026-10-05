# Dataset / Track API prototype

**Throwaway experiment, not the production pbzarr API.** This explores the
[proposed core convention](../../../docs/proposals/format-05/README.md).
It does not read or migrate production format 0.4, and does not change its
implementation or release scope.

The question: can friendly Dataset and Track classes coexist with native
Xarray without implementing another array library? This prototype combines
small explicit adapters, a real custom genomic index, and shared genomic
helpers. The point is to assess the interface and its actual maintenance cost.

## Run

From the repository root, using the existing environment:

```sh
pixi run dataset-track-poc
pixi run dataset-track-poc --interactive
```

The first command demonstrates the complete workflow and checks its results.
All stores are temporary and removed afterward. Interactive mode is an
in-memory playground for selection, callable counts, native quantiles, and
coarsening; it has no persistent writes.

If the execution environment prevents Pixi from taking its global cache lock,
the equivalent command uses the installed project interpreter directly:

```sh
.pixi/envs/default/bin/python python/prototypes/dataset_track_poc/demo.py
```

Nothing needs to be installed. For import into a scratch notebook or Python
session, put `python/prototypes` on its import path and import
`Dataset` and `Track` from `dataset_track_poc`. This package is deliberately
separate from `pbzarr`.

## The implemented class surface

The complete implementations are in [api.py](api.py). This is their public
shape; abbreviated signatures below are a reading guide.

```python
class Dataset:
    @classmethod
    def open(cls, source, *, mode="r", chunks="auto"): ...
    @classmethod
    def create(cls, destination, *, reference, coords=None, attrs=None): ...
    @classmethod
    def from_xarray(cls, dataset): ...

    # reference, tracks, coords, attrs, sizes
    def __getitem__(self, name) -> Track: ...
    def select(self, *, region=None, **labels) -> Dataset: ...
    def assign(self, **tracks) -> Dataset: ...
    def write(self, tracks, *, encoding=None) -> None: ...
    def to_xarray(self): ...
    def compute(self, **kwargs) -> Dataset: ...
    def close(self) -> None: ...
    # Context manager support.


class Track:
    @classmethod
    def from_array(cls, data, *, dims, coords=None, reference=None, attrs=None): ...
    @classmethod
    def from_xarray(cls, array, *, reference=None): ...

    # dims, shape, sizes, dtype, coords, attrs, reference, index
    def select(self, *, region=None, **labels) -> Track: ...
    def sum(self, dim=None, **kwargs) -> Track: ...
    def mean(self, dim=None, **kwargs) -> Track: ...
    def min(self, dim=None, **kwargs) -> Track: ...
    def max(self, dim=None, **kwargs) -> Track: ...
    def any(self, dim=None, **kwargs) -> Track: ...
    def all(self, dim=None, **kwargs) -> Track: ...
    def astype(self, dtype, **kwargs) -> Track: ...
    def with_attrs(self, **attrs) -> Track: ...
    def coarsen(self, *, position): ...
    def to_xarray(self): ...
    def compute(self, **kwargs) -> Track: ...
    def item(self): ...
    # Comparisons, +, -, *, /, &, |, ~; no blanket __getattr__ proxy.
```

`Dataset` wraps an actual `xr.Dataset`. `Track` wraps an actual `xr.DataArray`.
Both retain a small immutable `Reference`. The custom `GenomicIndex` lives in
Xarray's own index collection and carries reference and row identity through
supported native operations. There is no public `geometry` argument.

Dataset/Track selection is functional: it returns a new view. `assign` is
in-memory composition with exact matching axes; it never writes or scatters.
Only a root opened with `mode="r+"`, or returned by `create`, can write. Derived
views do not silently acquire write authority. User attributes are attached
with `with_attrs`; this prototype omits in-place metadata update methods.

## What implementing sum actually involves

This is the real adapter:

```python
def sum(self, dim=None, **kwargs):
    return self._reduce("sum", dim, **kwargs)

def _reduce(self, method, dim=None, **kwargs):
    self._check_open()
    kwargs.setdefault("keep_attrs", False)
    return self._wrap(getattr(self._array, method)(dim=dim, **kwargs))
```

The shared result wrapper checks the surviving index, normalizes the genomic
axis to first position, and retains the owner's lifetime. Xarray handles the
actual calculation, missing values, dtype behavior, and Dask execution.
The existing native index follows the result when the genomic axis survives;
reducing that axis removes it. There is no operation graph or provenance engine.

Binary adapters share a separate helper that validates exact genomic identity
and sample/other dimension labels before calling Xarray. That alignment policy
is substantive PBZ behavior; the arithmetic itself is one delegated operation.

The maintenance cost is therefore mostly in reference/index validity,
alignment policy, lifetime, and writing. Adding another named reducer is small.
Trying to transparently proxy every native method would make the contract much
larger, so unsupported operations cross the explicit Xarray boundary.

## Both styles work

```python
# Wrapper style.
depth = ds["depth"].select(region="chr1:100-1000")
counts = (depth >= 5).sum("sample").with_attrs(notes="Depth >= 5")

# Native style, using the same custom index.
native = ds["depth"].to_xarray().sel(position="chr1:100-1000")
native_counts = (native >= 5).sum("sample")
counts = Track.from_xarray(native_counts)

# No Track.quantile method is necessary.
quantiles = Track.from_xarray(
    native.quantile([0.25, 0.75], dim="sample")
)
```

An indexed native array carries enough context to wrap or write it. A genomic
array with its index removed raises an error; shape and inherited attributes
cannot restore its meaning. `Track.from_array(..., reference=...)` is an
explicit declaration of canonical full-reference order for external input.

`to_xarray()` shares underlying measurement buffers while copying the container.
It is not a deep immutability boundary. Arbitrary mutation of native arrays is
outside the wrapper guarantee. Native escape also gives up automatic owner-close
checks; finish native calculations before closing their source.

## Shared genomic helpers

```python
means = ds["depth"].coarsen(position=1000).mean()

# Same implementation on native Xarray.
native_means = ds["depth"].to_xarray().pbz_poc.coarsen(position=1000).mean()
```

The experimental accessor is named `pbz_poc` to avoid modifying the installed
`.pbz` accessor. A future adopted version would use the real accessor.

[coarsen.py](coarsen.py) delegates full windows and clipped terminal bins to
Xarray. It adds contig boundaries and a `bin_N` output index. It does not implement
numerical reduction kernels. In the example, contigs of lengths 1500 and 2200
produce five 1000-base bins, with terminal lengths 500 and 200.

The prototype requires complete, canonically ordered base data for coarsening.
It rejects regional inputs, reversed inputs, empty references, and zero-length
contigs for this operation. Plain native coarsening/rolling still has ordinary
array semantics; the custom index does not change window planning.

## Writing

```python
with Dataset.create("new_results.pbz", reference=ds.reference) as output:
    output.write({
        "counts": (ds["depth"] >= 5).sum("sample"),
        "mean_depth_1000": ds["depth"].coarsen(position=1000).mean(),
    })

    regional = (ds["depth"].select(region="chr1:100-1000") >= 5).sum("sample")
    output.write(
        {"regional_counts": regional},
        encoding={"regional_counts": {"fill_value": 0}},
    )
```

New variables accept either Tracks or indexed native DataArrays. The writer
checks reference/labels, prevents name collisions, and removes runtime genomic
coordinates. The native store retains the core draft's implicit genomic axes.
Regional results create full-reference destination shapes with explicit fill
elsewhere. Writes use lazy arrays and retain Xarray's chunk-safety checks.

This writer is a feasibility example, not a publication protocol. It supports
local Zarr v3, fixed reference array names, and contiguous increasing genomic
selections. Shared non-genomic labels must match exactly. It omits selected-column
scatter, append, replacement, patching, transaction recovery, and concurrent
writers. Regional writes reject genomic auxiliary coordinates and misaligned
explicit chunks. Its default chunk choices are for correctness in this example,
not measured performance recommendations. No Rust interoperability claim follows
from the native Xarray round trip.

## Observations from running it

Checked with the existing project interpreter: Python 3.12, Xarray 2025.12.0,
Dask 2026.8.0, Zarr 3.2.1. The demo completed successfully. Pixi's task entry was
added, but `pixi run` in the sandbox could not write its global cache lock; the
equivalent installed-interpreter command was used. No dependencies were installed.

The demo verified:

* Wrapper counts, threshold broadcasting, native counts, and contig-bin means
  agree with independent NumPy calculations on synthetic data.
* The native quantile result can return to Track without a context keyword.
* Constructing selected views, counts, quantiles, and bins executed zero Dask
  tasks. Measurement outputs remained lazy.
* Native reversed slicing preserves reversed genomic identity. Mismatched
  genomic/sample order and missing indexes raise errors.
* Full and regional writes reopen through plain Xarray with correct values,
  reference coordinates, fill, and user notes. Virtual genomic coordinates are
  absent on disk and reconstructed by the prototype opener.
* Computed scalar results remain usable after closing their source; an
  uncomputed Track detects a closed owner.

Additional review checks verified both owners are retained when combining
Tracks from separate Datasets, coarsening clears derived attributes by default,
coordinate mappings reject replacement, and displaying a `chunks=None` Dataset
does not load its measurement arrays.

A three-billion-row coordinate is represented with `pandas.RangeIndex`, reporting
136 bytes of index storage. This is a compact-coordinate demonstration, not
total memory usage, a whole-genome task-graph benchmark, or a read benchmark.

GPU backends are not implemented or tested. Data wrappers avoid implicit
measurement-to-NumPy conversion, but that alone does not establish GPU support.
The custom index supports slices, including reverse/strided slices; fancy
indexing, arbitrary genomic joins, concatenation, and renaming are deliberately
unsupported. The index uses an Xarray internal import for `IndexSelResult`,
and the scalar materialization check also uses a private Xarray attribute.
These need a supported-version policy before production use.

Keep this prototype only while evaluating the interface. If the design is
adopted, implement the chosen contract in the production package with its own
compatibility and validation work, then remove this experiment.
