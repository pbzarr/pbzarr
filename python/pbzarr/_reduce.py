"""Segmented region reduction: reduce a track over many disjoint intervals into a
(region x column) matrix.

The reduction is a groupby over the flat position axis: every position is labelled
with the interval id it falls in (or -1 for "no interval"), then flox reduces the
position axis grouped by that label. Intervals are culled to the chunks they touch
so a scattered million-interval query never reads untouched chunks.

Eager (numpy/cupy) path only for now; the dask construction reuses these helpers.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

from ._read import _TrackArrays, open_track
from ._region import RegionQuery

# ADR 0002: reductions are nan-aware. Map the public reducer to flox's nan-variant
# so a NaN fill (bigWig gap) is skipped while a numeric fill (d4 zero) participates.
_FLOX_FUNC = {
    "mean": "nanmean",
    "sum": "nansum",
    "count": "count",
    "min": "nanmin",
    "max": "nanmax",
    "var": "nanvar",
    "std": "nanstd",
}


def _columnar_intervals(intervals):
    """Return (contig_names, starts, ends) arrays for a columnar input, else None.

    Columnar inputs avoid a Python object per interval: a pandas DataFrame with
    contig/start/end columns, or a 3-tuple of equal-length array-likes
    `(contigs, starts, ends)`. The per-tuple form (a list of `(c, s, e)`) is handled
    by the scalar path.
    """
    if hasattr(intervals, "columns") and hasattr(intervals, "to_numpy"):  # DataFrame
        cols = {str(c).lower().lstrip("#"): c for c in intervals.columns}
        contig_col = next((cols[k] for k in ("contig", "chrom", "chromosome", "seqname") if k in cols), None)
        if contig_col is None or "start" not in cols or "end" not in cols:
            raise ValueError("interval DataFrame needs contig/chrom, start, and end columns")
        return (intervals[contig_col].to_numpy(),
                intervals[cols["start"]].to_numpy(),
                intervals[cols["end"]].to_numpy())
    if isinstance(intervals, tuple) and len(intervals) == 3 and not isinstance(intervals[0], str) \
            and hasattr(intervals[0], "__len__"):
        return intervals
    return None


def _normalize_intervals(intervals, contigs: list[str]):
    """Turn the interval query into parallel (contig_id, start, end) int arrays.

    Accepts a pandas DataFrame or a 3-tuple of arrays `(contigs, starts, ends)` (the
    scalable, vectorized path), or an iterable of `(contig, start, end)` / RegionQuery
    (fine for small N). Contig names resolve to ids against the track's contigs.
    """
    contig_index = {name: i for i, name in enumerate(contigs)}

    columnar = _columnar_intervals(intervals)
    if columnar is not None:
        names, starts, ends = columnar
        import pandas as pd

        ids = pd.Series(np.asarray(names).astype(str)).map(contig_index)
        if ids.isna().any():
            missing = sorted(set(np.asarray(names).astype(str)) - set(contig_index))
            raise KeyError(f"contig {missing[0]!r} not in track; contigs: {contigs}")
        return (ids.to_numpy().astype(np.int64),
                np.asarray(starts, dtype=np.int64),
                np.asarray(ends, dtype=np.int64))

    names_l: list[str] = []
    starts_l: list[int] = []
    ends_l: list[int] = []
    for item in intervals:
        if isinstance(item, RegionQuery):
            name, start, end = item.contig, item.start, item.end
        else:
            name, start, end = item[0], item[1], item[2]
        if start is None or end is None:
            raise ValueError(f"interval {item!r} needs explicit start and end")
        names_l.append(str(name))
        starts_l.append(int(start))
        ends_l.append(int(end))

    try:
        contig_ids = np.array([contig_index[n] for n in names_l], dtype=np.int64)
    except KeyError as e:
        raise KeyError(f"contig {e.args[0]!r} not in track; contigs: {contigs}") from e
    return contig_ids, np.asarray(starts_l, dtype=np.int64), np.asarray(ends_l, dtype=np.int64)


def compute_boundaries(contig_ids, starts, ends, offsets):
    """Resolve intervals to sorted, disjoint flat spans; keep original input order.

    Returns (sorted_starts, sorted_ends, interval_ids), where interval_ids[j] is the
    original input index of the j-th sorted interval, so grouping by it yields output
    rows in input order.
    """
    offsets = np.asarray(offsets)
    flat_starts = offsets[contig_ids] + starts
    contig_lengths = offsets[contig_ids + 1] - offsets[contig_ids]
    flat_ends = offsets[contig_ids] + np.minimum(ends, contig_lengths)

    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("each interval must satisfy 0 <= start < end")

    sort_order = np.argsort(flat_starts, kind="stable")
    sorted_starts = flat_starts[sort_order]
    sorted_ends = flat_ends[sort_order]
    interval_ids = sort_order.astype(np.int32)

    overlaps = sorted_starts[1:] < sorted_ends[:-1]

    if np.any(overlaps):
        i = np.flatnonzero(overlaps)[0]
        a = interval_ids[i]
        b = interval_ids[i + 1]

        raise ValueError(
            "Overlap:\n"
            f"{contig_ids[a]=}, {starts[a]=}, {ends[a]=}, "
            f"flat=[{sorted_starts[i]}, {sorted_ends[i]})\n"
            f"{contig_ids[b]=}, {starts[b]=}, {ends[b]=}, "
            f"flat=[{sorted_starts[i + 1]}, {sorted_ends[i + 1]})"
        )

    return sorted_starts, sorted_ends, interval_ids


def labels_for_positions(positions, sorted_starts, sorted_ends, interval_ids, array_ns):
    """Original interval id per flat position, or -1 in a gap. Backend-generic."""
    stab = array_ns.searchsorted(sorted_starts, positions, side="right") - 1
    stab_safe = array_ns.clip(stab, 0, len(sorted_starts) - 1)
    inside = (stab >= 0) & (positions < sorted_ends[stab_safe])
    return array_ns.where(inside, interval_ids[stab_safe], -1).astype(np.int32)


def coalesce_touched_chunks(sorted_starts, sorted_ends, chunk_width, total_len):
    """Chunk-aligned position slabs covering every touched chunk, adjacent ones merged."""
    first = sorted_starts // chunk_width
    last = (sorted_ends - 1) // chunk_width
    touched = np.unique(np.concatenate([np.arange(f, l + 1) for f, l in zip(first, last)]))

    slabs: list[tuple[int, int]] = []
    run_start = touched[0]
    prev = touched[0]
    for chunk in touched[1:]:
        if chunk != prev + 1:
            slabs.append((int(run_start * chunk_width), int(min((prev + 1) * chunk_width, total_len))))
            run_start = chunk
        prev = chunk
    slabs.append((int(run_start * chunk_width), int(min((prev + 1) * chunk_width, total_len))))
    return slabs


def _reduce_eager(ta: _TrackArrays, reduce: str, column, contig_ids, starts, ends):
    import flox.xarray

    func = _FLOX_FUNC[reduce]
    n_intervals = len(contig_ids)

    is_2d = ta.col_dim is not None
    col_idx = None
    if column is not None:
        if not is_2d:
            raise ValueError("column= is only valid on a 2D track")
        col_idx = ta.labels.index(column)

    sorted_starts, sorted_ends, interval_ids = compute_boundaries(
        contig_ids, starts, ends, ta.offsets
    )
    chunk_width = ta.values.chunks[0] if hasattr(ta.values, "chunks") else ta.values.shape[0]
    total_len = ta.values.shape[0]
    slabs = coalesce_touched_chunks(sorted_starts, sorted_ends, chunk_width, total_len)

    value_slabs = []
    label_slabs = []
    array_ns = np
    for slab_start, slab_end in slabs:
        if is_2d and col_idx is None:
            slab_values = np.asarray(ta.values[slab_start:slab_end, :])
        elif is_2d:
            slab_values = np.asarray(ta.values[slab_start:slab_end, col_idx])
        else:
            slab_values = np.asarray(ta.values[slab_start:slab_end])
        array_ns = np  # cupy dispatch lands here once GPU path is wired
        slab_labels = labels_for_positions(
            array_ns.arange(slab_start, slab_end),
            sorted_starts, sorted_ends, interval_ids, array_ns,
        )
        value_slabs.append(slab_values)
        label_slabs.append(slab_labels)

    all_values = array_ns.concatenate(value_slabs, axis=0)
    all_labels = array_ns.concatenate(label_slabs, axis=0)

    reduced_dims = ("position", ta.col_dim) if (is_2d and col_idx is None) else ("position",)
    values_da = xr.DataArray(all_values, dims=reduced_dims)
    by_da = xr.DataArray(all_labels, dims="position", name="region")
    result = flox.xarray.xarray_reduce(
        values_da, by_da, func=func, dim="position",
        expected_groups=np.arange(n_intervals), fill_value=np.nan,
    )
    return _finish(result, ta, column, contig_ids, starts, ends)


def build_labels_dask(slabs, value_slabs, sorted_starts, sorted_ends, interval_ids):
    """Concatenated dask label array aligned to the value slabs' position chunks.

    The boundary arrays are closed over by `label_block`; dask/cloudpickle stores that
    function once per graph (not per block), so the boundaries are serialized a single
    time regardless of block count. For a distributed run with a huge interval count,
    `client.scatter` the boundaries so that single copy is broadcast and cached rather
    than shipped inside each submitted graph.
    """
    import dask.array as da

    def label_block(positions):
        return labels_for_positions(positions, sorted_starts, sorted_ends, interval_ids, np)

    label_slabs = []
    for (slab_start, slab_end), slab_values in zip(slabs, value_slabs):
        positions = da.arange(slab_start, slab_end, chunks=slab_values.chunks[0])
        label_slabs.append(da.map_blocks(label_block, positions, dtype=np.int32))
    return da.concatenate(label_slabs, axis=0)


def _col_index(ta: _TrackArrays, column):
    if column is None:
        return None
    if ta.col_dim is None:
        raise ValueError("column= is only valid on a 2D track")
    return ta.labels.index(column)


def _reduce_dask(ta: _TrackArrays, reduce: str, column, contig_ids, starts, ends):
    import dask.array as da
    import flox.xarray

    func = _FLOX_FUNC[reduce]
    n_intervals = len(contig_ids)
    is_2d = ta.col_dim is not None
    col_idx = _col_index(ta, column)

    sorted_starts, sorted_ends, interval_ids = compute_boundaries(
        contig_ids, starts, ends, ta.offsets
    )
    values = ta.values if _is_dask(ta.values) else da.from_array(ta.values, chunks=ta.values.chunks)
    chunk_width = values.chunks[0][0]
    total_len = values.shape[0]
    slabs = coalesce_touched_chunks(sorted_starts, sorted_ends, chunk_width, total_len)

    value_slabs = []
    for slab_start, slab_end in slabs:
        if is_2d and col_idx is None:
            value_slabs.append(values[slab_start:slab_end, :])
        elif is_2d:
            value_slabs.append(values[slab_start:slab_end, col_idx])
        else:
            value_slabs.append(values[slab_start:slab_end])

    all_values = da.concatenate(value_slabs, axis=0)
    # cohorts needs a numpy grouper; labels are cheap searchsorted output, so materialize
    # them while the value array stays lazy/streamed.
    all_labels = np.asarray(
        build_labels_dask(slabs, value_slabs, sorted_starts, sorted_ends, interval_ids)
    )

    reduced_dims = ("position", ta.col_dim) if (is_2d and col_idx is None) else ("position",)
    values_da = xr.DataArray(all_values, dims=reduced_dims)
    by_da = xr.DataArray(all_labels, dims="position", name="region")
    result = flox.xarray.xarray_reduce(
        values_da, by_da, func=func, dim="position",
        expected_groups=np.arange(n_intervals), fill_value=np.nan,
        method="cohorts",
    )
    return _finish(result, ta, column, contig_ids, starts, ends)


def _is_dask(obj) -> bool:
    return type(obj).__module__.startswith("dask.")


def _finish(result, ta: _TrackArrays, column, contig_ids, starts, ends):
    lead = ("region", ta.col_dim) if (ta.col_dim is not None and column is None) else ("region",)
    result = result.transpose(*lead)
    result = result.assign_coords(
        region=np.arange(len(contig_ids)),
        region_contig=("region", np.array([ta.contigs[i] for i in contig_ids])),
        region_start=("region", np.asarray(starts)),
        region_end=("region", np.asarray(ends)),
    )
    if ta.col_dim is not None and column is None:
        result = result.assign_coords({ta.col_dim: ta.labels})
    return result


def _wrap(matrix, ta: _TrackArrays, column, col_idx, contig_ids, starts, ends):
    coords = {
        "region": np.arange(len(contig_ids)),
        "region_contig": ("region", np.array([ta.contigs[i] for i in contig_ids])),
        "region_start": ("region", np.asarray(starts)),
        "region_end": ("region", np.asarray(ends)),
    }
    if ta.col_dim is not None and column is None:
        return xr.DataArray(
            matrix, dims=("region", ta.col_dim),
            coords={**coords, ta.col_dim: ta.labels},
        )
    return xr.DataArray(matrix, dims=("region",), coords=coords)


def reduce_regions(store_path: str, name: str, intervals, reduce: str, column=None,
                   *, lazy: bool = False) -> xr.DataArray:
    if reduce not in _FLOX_FUNC:
        raise ValueError(f"unsupported reduce {reduce!r}; expected one of {sorted(_FLOX_FUNC)}")
    ta = open_track(store_path, name)
    contig_ids, starts, ends = _normalize_intervals(intervals, ta.contigs)
    if len(contig_ids) == 0:
        return _wrap(_empty_matrix(ta, column), ta, column, None, contig_ids, starts, ends)
    run = _reduce_dask if lazy else _reduce_eager
    return run(ta, reduce, column, contig_ids, starts, ends)


def _empty_matrix(ta: _TrackArrays, column):
    if ta.col_dim is not None and column is None:
        return np.empty((0, len(ta.labels)), dtype=np.float64)
    return np.empty((0,), dtype=np.float64)
