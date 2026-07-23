"""Dask path: correctness parity with the eager path, and graph/I-O optimization.

The optimization tests are the point here: build a case where a naive full-array
reduction touches every chunk, and assert our slab-culling reads only the touched
chunks (skipping untouched middle chunks) and yields a smaller task graph.
"""
from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

import flox.xarray

from pbzarr._read import _TrackArrays
from pbzarr._reduce import (
    _normalize_intervals,
    _reduce_dask,
    _reduce_eager,
    compute_boundaries,
    labels_for_positions,
)

CONTIGS = ["chr1"]
OFFSETS = np.array([0, 50], dtype=np.int64)
TOTAL = 50
CHUNK = 10           # 5 position chunks: [0,10) [10,20) [20,30) [30,40) [40,50)
NCOL = 2


class CountingArray:
    """zarr-like source that records which position-chunk each read hits."""

    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.ndim = arr.ndim
        self.reads: list[int] = []

    def __getitem__(self, key):
        pos = key[0] if isinstance(key, tuple) else key
        if isinstance(pos, slice) and pos.start is not None and (pos.stop is None or pos.stop > pos.start):
            self.reads.append(pos.start // CHUNK)
        return self._arr[key]


def _values():
    arr = np.empty((TOTAL, NCOL), dtype=np.float64)
    arr[:, 0] = np.arange(TOTAL)
    arr[:, 1] = 100 + np.arange(TOTAL)
    return arr


def _ta(values):
    return _TrackArrays(
        values=values, offsets=OFFSETS, contigs=CONTIGS,
        dims=("position", "sample"), col_dim="sample", labels=["s1", "s2"],
    )


def _ref(vals, intervals, reduce, col):
    fn = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}[reduce]
    return np.array([fn(vals[s:e, col], axis=0) for _, s, e in intervals])


def _naive_full_reduce(source, intervals):
    """Baseline we are avoiding: full array + full label vector, no slab culling."""
    values = da.from_array(source, chunks=(CHUNK, NCOL))
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    ss, se, ids = compute_boundaries(contig_ids, starts, ends, OFFSETS)
    full_by = labels_for_positions(np.arange(TOTAL), ss, se, ids, np)
    by_da = xr.DataArray(da.from_array(full_by, chunks=(CHUNK,)), dims="position", name="region")
    values_da = xr.DataArray(values, dims=("position", "sample"))
    return flox.xarray.xarray_reduce(
        values_da, by_da, func="nanmean", dim="position",
        expected_groups=np.arange(len(contig_ids)), fill_value=np.nan,
    )


@pytest.mark.parametrize("reduce", ["mean", "sum", "max"])
def test_dask_matches_eager_and_numpy(reduce):
    vals = _values()
    intervals = [("chr1", 2, 5), ("chr1", 22, 28), ("chr1", 42, 50)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)

    out = _reduce_dask(_ta(da.from_array(vals, chunks=(CHUNK, NCOL))), reduce, None, contig_ids, starts, ends)
    assert dask.is_dask_collection(out.data)          # lazy before compute

    computed = out.compute()
    eager = _reduce_eager(_ta(vals), reduce, None, contig_ids, starts, ends)
    assert computed.sel(sample="s1").values == pytest.approx(_ref(vals, intervals, reduce, 0).tolist())
    assert computed.values == pytest.approx(eager.values)


def test_column_selected_dask_matches_numpy():
    vals = _values()
    intervals = [("chr1", 0, 10), ("chr1", 40, 50)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    out = _reduce_dask(_ta(da.from_array(vals, chunks=(CHUNK, NCOL))), "mean", "s2", contig_ids, starts, ends)
    assert out.dims == ("region",)
    assert out.compute().values == pytest.approx(_ref(vals, intervals, "mean", 1).tolist())


def test_culling_reads_only_touched_chunks():
    # intervals hit chunk 0 and chunk 4; chunks 1,2,3 must never be read
    intervals = [("chr1", 2, 5), ("chr1", 42, 45)]
    source = CountingArray(_values())
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    _reduce_dask(_ta(da.from_array(source, chunks=(CHUNK, NCOL))), "mean", None, contig_ids, starts, ends).compute()
    assert set(source.reads) == {0, 4}


def test_naive_reads_all_chunks_but_culled_reads_few():
    intervals = [("chr1", 2, 5), ("chr1", 42, 45)]

    naive_src = CountingArray(_values())
    _naive_full_reduce(naive_src, intervals).compute()

    culled_src = CountingArray(_values())
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    _reduce_dask(_ta(da.from_array(culled_src, chunks=(CHUNK, NCOL))), "mean", None, contig_ids, starts, ends).compute()

    assert set(naive_src.reads) == {0, 1, 2, 3, 4}     # naive touches everything
    assert set(culled_src.reads) == {0, 4}             # culled touches only what it needs
    assert len(set(culled_src.reads)) < len(set(naive_src.reads))


def _reads_for_total(total_chunks, intervals):
    total = total_chunks * CHUNK
    offsets = np.array([0, total], dtype=np.int64)
    buf = np.empty((total, NCOL), dtype=np.float64)
    buf[:, 0] = np.arange(total)
    buf[:, 1] = 100 + np.arange(total)
    source = CountingArray(buf)
    ta = _TrackArrays(
        values=da.from_array(source, chunks=(CHUNK, NCOL)), offsets=offsets,
        contigs=CONTIGS, dims=("position", "sample"), col_dim="sample", labels=["s1", "s2"],
    )
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    _reduce_dask(ta, "mean", None, contig_ids, starts, ends).compute()
    return set(source.reads)


def test_culled_reads_scale_with_touched_not_total():
    # same intervals hit chunks 0 and 2 regardless of how large the array is
    intervals = [("chr1", 5, 8), ("chr1", 25, 28)]
    assert _reads_for_total(5, intervals) == {0, 2}
    assert _reads_for_total(50, intervals) == {0, 2}
    assert _reads_for_total(500, intervals) == {0, 2}


def _graph_size(collection):
    return len(dict(collection.__dask_graph__()))


def test_culled_task_graph_is_smaller():
    intervals = [("chr1", 2, 5), ("chr1", 42, 45)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)

    naive = _naive_full_reduce(CountingArray(_values()), intervals)
    culled = _reduce_dask(_ta(da.from_array(CountingArray(_values()), chunks=(CHUNK, NCOL))),
                          "mean", None, contig_ids, starts, ends)

    assert _graph_size(culled.data) < _graph_size(naive.data)
