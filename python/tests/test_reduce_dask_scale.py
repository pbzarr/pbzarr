"""Scale tests for the dask reduction.

The serialization test is the finding-turned-guard: `da.map_blocks` with a closure
serializes the boundary arrays ONCE per graph, not per block (verified with
cloudpickle, the distributed scheduler's serializer). This asserts that property so a
future refactor can't silently reintroduce per-block duplication.
"""
from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

from pbzarr._read import _TrackArrays
from pbzarr._reduce import _normalize_intervals, _reduce_dask, build_labels_dask


def test_boundaries_serialized_once_regardless_of_block_count():
    cloudpickle = pytest.importorskip("cloudpickle")
    n = 100_000
    rng = np.random.default_rng(0)
    sorted_starts = np.sort(rng.integers(0, 10_000_000, n))
    sorted_ends = sorted_starts + 1
    interval_ids = np.arange(n, dtype=np.int32)
    arrays_bytes = sorted_starts.nbytes + sorted_ends.nbytes + interval_ids.nbytes

    def copies(nblocks):
        chunk = 1000
        total = nblocks * chunk
        value_slabs = [da.zeros((total, 2), chunks=(chunk, 2))]
        by = build_labels_dask([(0, total)], value_slabs, sorted_starts, sorted_ends, interval_ids)
        size = len(cloudpickle.dumps(dict(by.__dask_graph__())))
        return size / arrays_bytes

    small, large = copies(20), copies(200)
    assert small < 2.0 and large < 2.0          # ~one copy of the boundaries, not per-block
    assert large < small * 1.5                  # stays ~constant as block count grows 10x


def test_many_disjoint_intervals_correctness():
    chunk, nchunks = 50, 40
    total = chunk * nchunks
    vals = np.empty((total, 2), dtype=np.float64)
    vals[:, 0] = np.arange(total)
    vals[:, 1] = np.arange(total) * 0.5
    ta = _TrackArrays(
        values=da.from_array(vals, chunks=(chunk, 2)),
        offsets=np.array([0, total], dtype=np.int64),
        contigs=["chr1"], dims=("position", "sample"), col_dim="sample", labels=["s1", "s2"],
    )
    starts = np.arange(0, total - 4, 6)          # width-4 intervals, gap 2 -> disjoint
    intervals = [("chr1", int(s), int(s + 4)) for s in starts]
    contig_ids, iv_starts, iv_ends = _normalize_intervals(intervals, ["chr1"])

    out = _reduce_dask(ta, "mean", None, contig_ids, iv_starts, iv_ends).compute()

    assert out.sizes["region"] == len(starts)
    ref0 = [vals[s:s + 4, 0].mean() for s in starts]
    ref1 = [vals[s:s + 4, 1].mean() for s in starts]
    assert out.sel(sample="s1").values == pytest.approx(ref0)
    assert out.sel(sample="s2").values == pytest.approx(ref1)
