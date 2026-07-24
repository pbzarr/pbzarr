"""The `.pbz` xarray Dataset accessor: region views + `top`.

`ds.pbz.regions(intervals)` culls a same-genome Dataset (from `store.dataset(...)`)
to the chunks its intervals touch and labels each position with its `region` (NaN
for gaps, so native `groupby("region")` drops them) and its global `flat_pos`. From
the returned view everything is ordinary xarray plus one selector, `top`.

Scale: the view is built by slab-slicing (chunk-aligned) + concat so the data stays
lazily chunk-backed, and the `region`/`flat_pos` coords are lazy dask arrays too, so
nothing genome-scale is materialized. `top` computes only the small per-region
reductions and broadcasts them back blockwise, so it stays bounded in memory over a
genome-scale view.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._reduce import (
    _FLOX_FUNC,
    _normalize_intervals,
    coalesce_touched_chunks,
    compute_boundaries,
    labels_for_positions,
)

_N_REGIONS_ATTR = "pbz_n_regions"


def _is_dask(obj) -> bool:
    return type(obj).__module__.startswith("dask.")


def _resolve_keys(by, view):
    items = by if isinstance(by, (list, tuple)) else [by]
    keys = []
    for item in items:
        if isinstance(item, str):
            keys.append(view[item])
        elif callable(item):
            keys.append(item(view))
        else:
            keys.append(item)
    return keys


def _merged_spans(sorted_starts, sorted_ends):
    """Merge sorted disjoint spans into covered runs (numpy-backing slabs)."""
    slabs = []
    cur_s, cur_e = int(sorted_starts[0]), int(sorted_ends[0])
    for a, b in zip(sorted_starts[1:].tolist(), sorted_ends[1:].tolist()):
        if a <= cur_e:
            cur_e = max(cur_e, b)
        else:
            slabs.append((cur_s, cur_e))
            cur_s, cur_e = a, b
    slabs.append((cur_s, cur_e))
    return slabs


def _slab_labels(a, b, chunk_width, sorted_starts, sorted_ends, interval_ids):
    """(region float with NaN gaps, flat_pos int) for a slab; lazy when chunked."""
    def block(positions):
        lab = labels_for_positions(positions, sorted_starts, sorted_ends, interval_ids, np)
        return np.where(lab < 0, np.nan, lab.astype(np.float64))

    if chunk_width is not None:
        import dask.array as da

        pos = da.arange(a, b, chunks=chunk_width)
        return da.map_blocks(block, pos, dtype=np.float64), pos
    pos = np.arange(a, b)
    return block(pos), pos


def _broadcast_per_region(per_region: xr.DataArray, region_da: xr.DataArray) -> xr.DataArray:
    """Spread a per-region reduction back over positions, NaN at gaps, staying lazy.

    Only the small per-region array is materialized; the position-length result is a
    blockwise map over the (possibly dask) region-id coordinate.
    """
    per_region = per_region.transpose("region", ...)
    vals = np.asarray(per_region.values)               # (n_regions, *rest) -- small
    rest_shape = vals.shape[1:]
    rest_dims = tuple(d for d in per_region.dims if d != "region")

    def block(region_block):
        safe = np.where(np.isnan(region_block), 0, region_block).astype(np.int64)
        out = vals[safe]                               # (len, *rest)
        out[np.isnan(region_block)] = np.nan
        return out

    data = region_da.data
    if _is_dask(data):
        import dask.array as da

        n_extra = len(rest_shape)
        out = da.map_blocks(
            block, data, dtype=np.float64,
            new_axis=list(range(1, 1 + n_extra)),
            chunks=(data.chunks[0],) + rest_shape,
        )
    else:
        out = block(np.asarray(data))
    return xr.DataArray(out, dims=("position",) + rest_dims)


@xr.register_dataset_accessor("pbz")
class PbzAccessor:
    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        self._region_by = None

    def _region_grouper(self):
        """Region labels as a numpy-backed DataArray, computed once and reused.

        flox `method="cohorts"` inspects the group->chunk map up front and rejects a
        dask grouper, so the labels must be numpy. They are cheap searchsorted output;
        materializing this one (position,) array keeps the value variables lazy while
        avoiding the map-reduce densification. `top()` drives several reductions off the
        same labels, so cache it on the accessor.
        """
        if self._region_by is None:
            self._region_by = self._ds["region"].compute()
        return self._region_by

    def regions(self, intervals) -> xr.Dataset:
        """Cull to the touched chunks and label `region` (NaN gaps) + `flat_pos`."""
        ds = self._ds
        offsets = np.asarray(ds["offsets"].values)
        contigs = [str(c) for c in np.asarray(ds["contigs"].values)]
        contig_ids, starts, ends = _normalize_intervals(intervals, contigs)
        sorted_starts, sorted_ends, interval_ids = compute_boundaries(
            contig_ids, starts, ends, offsets
        )
        total = int(ds.sizes["position"])

        chunks = ds.chunksizes.get("position")
        chunk_width = chunks[0] if chunks else None
        if chunk_width is not None:
            slabs = coalesce_touched_chunks(sorted_starts, sorted_ends, chunk_width, total)
        else:
            slabs = _merged_spans(sorted_starts, sorted_ends)

        parts = []
        for a, b in slabs:
            sub = ds.isel(position=slice(a, b))
            region, flat = _slab_labels(a, b, chunk_width, sorted_starts, sorted_ends, interval_ids)
            parts.append(sub.assign_coords(region=("position", region), flat_pos=("position", flat)))
        view = parts[0] if len(parts) == 1 else xr.concat(
            parts, dim="position", coords="minimal", compat="override", combine_attrs="override"
        )
        view.attrs[_N_REGIONS_ATTR] = int(len(contig_ids))
        return view

    def _n_regions(self) -> int:
        n = self._ds.attrs.get(_N_REGIONS_ATTR)
        if n is None:
            raise ValueError("not a region view; call .pbz.regions(intervals) first")
        return int(n)

    def _reduce(self, obj, func, fill_value=np.nan):
        """flox segmented reduction over the region label; lazy over a dask grouper.

        Drops the genome coords (`contigs`/`offsets`) so the per-region result has only
        the `region` (and any column) dims and does not cross-product in `to_dataframe`.

        `method="cohorts"`: regions are disjoint and sorted along the flat position axis,
        so each block's labels are a small contiguous run of region ids. Cohorts sizes the
        per-block partial to the groups actually in that block; the default `map-reduce`
        densifies to all N regions per block (N × columns × 8B), which balloons to ~1 TB
        on a genome-scale view with millions of regions.
        """
        import flox.xarray

        result = flox.xarray.xarray_reduce(
            obj, self._region_grouper(), func=func, dim="position",
            expected_groups=np.arange(self._n_regions()), fill_value=fill_value,
            method="cohorts",
        )
        drop = [c for c in ("contigs", "offsets") if c in result.coords]
        return result.drop_vars(drop) if drop else result

    def reduce(self, func: str, *, fill_value=np.nan) -> xr.Dataset:
        """Per-region value reduction (mean/sum/min/max/...). Lazy on a dask view.

        Native `groupby("region")` needs eager labels; this passes them for you.
        """
        return self._reduce(self._ds, _FLOX_FUNC.get(func, func), fill_value=fill_value)

    def top(self, k: int = 1, *, by, descending=True, keep: str = "first",
            with_positions: bool = False) -> xr.Dataset:
        """Top-k rows per region ordered by `by` (top-1 only for now).

        Returns the winning rows (all data variables at the winning position per
        region). `by` is a name, DataArray, or callable(view), or an ordered list
        thereof (primary key first). Ties break by later keys, then position per `keep`.
        """
        if k != 1:
            raise NotImplementedError("top() supports k=1 only for now")
        view = self._ds
        self._n_regions()  # validates this is a region view
        region_da = view["region"]

        keys = _resolve_keys(by, view)
        descs = list(descending) if isinstance(descending, (list, tuple)) else [descending] * len(keys)

        cand = region_da.notnull()
        for key, desc in zip(keys, descs):
            ext = self._reduce(key.where(cand), "nanmax" if desc else "nanmin")
            cand = cand & (key == _broadcast_per_region(ext, region_da))

        fp = view["flat_pos"]
        winner = self._reduce(fp.where(cand), "nanmin" if keep == "first" else "nanmax")
        final = cand & (fp == _broadcast_per_region(winner, region_da))

        rows = self._reduce(view.where(final), "nanmax")
        if with_positions:
            rows = rows.assign_coords(flat_pos=winner)
        return rows
