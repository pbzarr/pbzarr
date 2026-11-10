"""Genome functions: operations on whole-genome xarray objects that need the reference."""

from __future__ import annotations

from functools import partial
from typing import TypeVar

import dask.array as da
import numpy as np
import xarray as xr

from . import _coord
from ._coord import _check_full
from ._locus import locus_coords
from ._native import PbzError
from ._reference import Reference, bin_width
from ._region import RegionQuery
from ._regions import interval_columns, reduce_regions_dataset

T = TypeVar("T", xr.DataArray, xr.Dataset)

_UNNAMED = "__pbzarr_values__"


def reduce_regions(
    obj: T,
    intervals,
    reducer: str,
    /,
    *,
    reference: Reference | None = None,
    **kwargs,
) -> T:
    """One statistic per interval, along ``position``, in one pass.

    ``obj`` is a whole-genome DataArray or Dataset. The result has the same
    type, with a ``region`` dimension in place of ``position``. On a Dataset,
    variables without ``position`` pass through unchanged. ``reference=``
    overrides the ``reference`` coordinate, or supplies it when ``obj`` has
    none. The result carries the reference that was used.
    """
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("reduce_regions needs an xarray DataArray or Dataset")
    ref = reference if reference is not None else Reference.from_xarray(obj)
    _check_full(obj, ref)
    if isinstance(obj, xr.Dataset):
        return _reduce_dataset(obj, ref, intervals, reducer, kwargs)
    if "position" not in obj.dims:
        raise ValueError("reduce_regions needs an array with a 'position' dimension")
    name = _UNNAMED if obj.name is None else obj.name
    result = _reduce_dataset(obj.to_dataset(name=name), ref, intervals, reducer, kwargs)[name]
    result.name = obj.name
    return result


def region(
    obj: T,
    query: str | RegionQuery | tuple[str, int | None, int | None],
    *,
    reference: Reference | None = None,
) -> T:
    """A view of one locus of a whole-genome DataArray or Dataset.

    Every genomic dimension present (``position`` and each ``bin_N``) is
    sliced to the locus, and variables with no genomic dimension pass through.
    The view carries the 0-d coordinates ``region_contig``, ``region_start``
    and ``region_end``. With ``position``, it also carries a numeric ``pos``
    coordinate and ``"contig:pos"`` labels as the ``position`` coordinate.
    Positions are 0-based and contig-local, like ``region_start``.
    """
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("region needs an xarray DataArray or Dataset")
    ref = reference if reference is not None else Reference.from_xarray(obj)
    _check_full(obj, ref)
    genomic = [dim for dim in obj.dims if bin_width(dim) is not None]
    if not genomic:
        raise ValueError("region needs a 'position' or 'bin_N' dimension")
    locus = ref.resolve(query)
    view = obj.isel({dim: ref.rows(locus, dim) for dim in genomic})
    coords = {
        "region_contig": ((), np.array(locus.contig, dtype=object)),
        "region_start": ((), locus.start),
        "region_end": ((), locus.end),
    }
    view = view.drop_vars(_coord.NAME, errors="ignore").assign_coords(coords)
    if "position" in view.dims:
        view = view.assign_coords(locus_coords(locus.contig, locus.start, locus.end))
    return _coord.attach(view, ref)


def interval_mask(
    obj: xr.DataArray | xr.Dataset,
    intervals,
    /,
    *,
    reference: Reference | None = None,
) -> xr.DataArray:
    """A lazy bool DataArray over ``position``: True inside any interval.

    ``obj`` is a whole-genome DataArray or Dataset with a ``position``
    dimension. ``intervals`` takes the forms of ``reduce_regions``; they are
    0-based and half-open, and they may overlap. The mask is a Dask array with
    the ``position`` chunks of ``obj``, and it carries the reference that was
    used.
    """
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("interval_mask needs an xarray DataArray or Dataset")
    if "position" not in obj.dims:
        raise ValueError("interval_mask needs an object with a 'position' dimension")
    ref = reference if reference is not None else Reference.from_xarray(obj)
    _check_full(obj, ref)

    contig_ids, starts, ends = interval_columns(ref, intervals)
    offsets = ref.offsets[contig_ids]
    starts, ends = _merged(offsets + starts, offsets + ends)
    chunks = _position_chunks(obj)
    if chunks is None:
        chunks = da.core.normalize_chunks("auto", (ref.size(),), dtype=np.bool_)[0]
    mask = da.map_blocks(
        partial(_mask_block, starts, ends),
        chunks=(chunks,),
        dtype=np.bool_,
        meta=np.empty(0, dtype=np.bool_),
    )
    return _coord.attach(xr.DataArray(mask, dims=("position",)), ref)


def _merged(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, disjoint, non-adjacent intervals that cover the same rows."""
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    reach = np.maximum.accumulate(ends[order])
    first = np.ones(starts.size, dtype=bool)
    first[1:] = starts[1:] > reach[:-1]
    last = np.append(np.flatnonzero(first)[1:] - 1, starts.size - 1)
    return starts[first], reach[last]


def _mask_block(starts: np.ndarray, ends: np.ndarray, block_info=None) -> np.ndarray:
    lo, hi = block_info[None]["array-location"][0]
    first = np.searchsorted(ends, lo, side="right")
    last = np.searchsorted(starts, hi, side="left")
    edges = np.zeros(hi - lo + 1, dtype=np.int8)
    edges[np.maximum(starts[first:last] - lo, 0)] = 1
    edges[np.minimum(ends[first:last] - lo, hi - lo)] = -1
    return np.cumsum(edges[:-1], dtype=np.int8) > 0


def _position_chunks(obj: xr.DataArray | xr.Dataset) -> tuple[int, ...] | None:
    variables = obj.variables.values() if isinstance(obj, xr.Dataset) else [obj.variable]
    for variable in variables:
        if "position" in variable.dims and variable.chunks is not None:
            return variable.chunks[variable.dims.index("position")]
    return None


def _reduce_dataset(
    ds: xr.Dataset, reference: Reference, intervals, reducer: str, kwargs: dict
) -> xr.Dataset:
    tracks = [name for name, value in ds.data_vars.items() if "position" in value.dims]
    if not tracks:
        raise ValueError("reduce_regions needs at least one variable with a 'position' dimension")
    for name in tracks:
        if ds[name].ndim > 2:
            raise PbzError(
                f"track {name!r} has {ds[name].ndim} dimensions; region operations "
                "handle position plus at most one more"
            )
    work = ds[tracks]
    work = work.drop_vars(
        [name for name, coord in work.coords.items() if "position" in coord.dims]
    )
    # The planner needs position first. A transposed array loses the fast path.
    work = work.assign(
        {
            name: value.transpose("position", ...)
            for name, value in work.data_vars.items()
            if value.dims[0] != "position"
        }
    )
    reduced = reduce_regions_dataset(work, reference, intervals, reducer, **kwargs)

    rest = ds.drop_vars([*tracks, _coord.NAME], errors="ignore")
    rest = rest.drop_vars([name for name, coord in rest.coords.items() if "position" in coord.dims])
    out = rest.drop_attrs(deep=False)
    for name in tracks:
        out[name] = reduced[name].variable
    out = out.assign_coords(
        {name: coord.variable for name, coord in reduced.coords.items() if name not in out.coords}
    )
    out = _coord.attach(out, reference)
    out.set_close(ds.close)
    return out
