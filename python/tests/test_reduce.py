"""Unit tests for the segmented region reduction core (eager path).

Store-free: builds `_TrackArrays` over an in-memory values array so these run
without cargo or htslib.
"""
from __future__ import annotations

import numpy as np
import pytest

from pbzarr._read import _TrackArrays
from pbzarr._reduce import (
    _normalize_intervals,
    _reduce_eager,
    coalesce_touched_chunks,
    compute_boundaries,
    labels_for_positions,
)

CONTIGS = ["chr1", "chr2"]
LENGTHS = [10, 8]
OFFSETS = np.array([0, 10, 18], dtype=np.int64)
TOTAL = 18


class _ChunkedNumpy:
    """Minimal zarr-like: exposes `.chunks`/`.shape` and slicing over a numpy buffer."""

    def __init__(self, arr, chunk0):
        self._arr = arr
        self.shape = arr.shape
        self.chunks = (chunk0,) + arr.shape[1:]

    def __getitem__(self, key):
        return self._arr[key]


def _cohort_values():
    vals = np.empty((TOTAL, 2), dtype=np.float64)
    vals[:, 0] = np.arange(TOTAL)
    vals[:, 1] = 100 + np.arange(TOTAL)
    return vals


def _ta(values, *, cohort=True):
    return _TrackArrays(
        values=values,
        offsets=OFFSETS,
        contigs=CONTIGS,
        dims=("position", "sample") if cohort else ("position",),
        col_dim="sample" if cohort else None,
        labels=["s1", "s2"] if cohort else None,
    )


def _ref(vals, intervals, reduce, col=None):
    fn = {"mean": np.nanmean, "sum": np.nansum, "min": np.nanmin, "max": np.nanmax}[reduce]
    rows = []
    for contig, start, end in intervals:
        base = OFFSETS[CONTIGS.index(contig)]
        seg = vals[base + start : base + end]
        seg = seg if col is None else seg[:, col]
        rows.append(fn(seg, axis=0))
    return np.asarray(rows)


def test_compute_boundaries_sorts_and_keeps_input_order():
    contig_ids, starts, ends = _normalize_intervals(
        [("chr2", 2, 5), ("chr1", 0, 4)], CONTIGS
    )
    ss, se, ids = compute_boundaries(contig_ids, starts, ends, OFFSETS)
    assert ss.tolist() == [0, 12]           # sorted by flat start
    assert se.tolist() == [4, 15]
    assert ids.tolist() == [1, 0]           # original input index of each sorted interval


def test_compute_boundaries_rejects_overlap():
    contig_ids, starts, ends = _normalize_intervals([("chr1", 0, 5), ("chr1", 3, 8)], CONTIGS)
    with pytest.raises(ValueError, match="disjoint"):
        compute_boundaries(contig_ids, starts, ends, OFFSETS)


def test_compute_boundaries_rejects_reversed():
    contig_ids, starts, ends = _normalize_intervals([("chr1", 5, 2)], CONTIGS)
    with pytest.raises(ValueError, match="start < end"):
        compute_boundaries(contig_ids, starts, ends, OFFSETS)


def test_labels_for_positions_marks_gaps():
    ss = np.array([0, 12])
    se = np.array([4, 15])
    ids = np.array([1, 0], dtype=np.int32)
    labels = labels_for_positions(np.arange(18), ss, se, ids, np)
    # [0,4) -> id 1 ; [4,12) gap ; [12,15) -> id 0 ; [15,18) gap
    assert labels[:4].tolist() == [1, 1, 1, 1]
    assert labels[4:12].tolist() == [-1] * 8
    assert labels[12:15].tolist() == [0, 0, 0]
    assert labels[15:].tolist() == [-1, -1, -1]


def test_coalesce_touched_chunks_merges_adjacent_and_skips_gaps():
    ss = np.array([3, 15])
    se = np.array([8, 18])       # chunks 0,1 (width 5) touched; chunk 3 touched; chunk 2 skipped
    slabs = coalesce_touched_chunks(ss, se, chunk_width=5, total_len=TOTAL)
    assert slabs == [(0, 10), (15, 18)]


@pytest.mark.parametrize("reduce", ["mean", "sum", "min", "max"])
def test_cohort_matrix_matches_numpy(reduce):
    vals = _cohort_values()
    ta = _ta(vals)
    intervals = [("chr1", 0, 4), ("chr2", 2, 5), ("chr1", 6, 10)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    out = _reduce_eager(ta, reduce, None, contig_ids, starts, ends)
    assert out.dims == ("region", "sample")
    assert out.sel(sample="s1").values.tolist() == pytest.approx(_ref(vals, intervals, reduce, 0).tolist())
    assert out.sel(sample="s2").values.tolist() == pytest.approx(_ref(vals, intervals, reduce, 1).tolist())


def test_column_selection_reduces_to_1d():
    vals = _cohort_values()
    ta = _ta(vals)
    intervals = [("chr1", 0, 4), ("chr1", 6, 10)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    out = _reduce_eager(ta, "mean", "s2", contig_ids, starts, ends)
    assert out.dims == ("region",)
    assert out.values.tolist() == pytest.approx(_ref(vals, intervals, "mean", 1).tolist())


def test_scalar_track_reduces_to_1d():
    vals = np.arange(TOTAL, dtype=np.float64)
    ta = _ta(vals, cohort=False)
    intervals = [("chr1", 1, 5), ("chr2", 0, 3)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    out = _reduce_eager(ta, "sum", None, contig_ids, starts, ends)
    assert out.dims == ("region",)
    assert out.values.tolist() == pytest.approx(_ref(vals[:, None], intervals, "sum", 0).tolist())


def test_chunk_spanning_and_culling():
    vals = _cohort_values()
    ta = _ta(_ChunkedNumpy(vals, chunk0=5))
    # first interval straddles the chunk-0/1 boundary; second lives in the last chunk only
    intervals = [("chr1", 3, 8), ("chr2", 5, 8)]
    contig_ids, starts, ends = _normalize_intervals(intervals, CONTIGS)
    out = _reduce_eager(ta, "mean", None, contig_ids, starts, ends)
    assert out.sel(sample="s1").values.tolist() == pytest.approx(_ref(vals, intervals, "mean", 0).tolist())
