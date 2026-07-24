"""Barebones tests for the .pbz accessor: regions() view and top()."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pbzarr  # noqa: F401 - registers the .pbz accessor

CONTIGS = ["chr1", "chr2"]
OFFSETS = [0, 10, 18]
TOTAL = 18


def make_ds(chunks=None):
    rng = np.arange(TOTAL)
    fire = np.stack([rng % 7, (rng * 2) % 5], axis=1).astype(np.float64)      # (pos, sample)
    cov = np.stack([10 + rng % 4, 20 + rng % 3], axis=1).astype(np.float64)
    ds = xr.Dataset(
        {
            "fire_coverage": (("position", "sample"), fire),
            "coverage": (("position", "sample"), cov),
        },
        coords={
            "sample": ["s1", "s2"],
            "contigs": ("contig", np.array(CONTIGS)),
            "offsets": ("contig_boundary", np.array(OFFSETS, dtype=np.int64)),
        },
    )
    return ds.chunk({"position": chunks}) if chunks else ds


def flat(contig, start):
    return OFFSETS[CONTIGS.index(contig)] + start


def test_regions_labels_and_culling():
    view = make_ds().pbz.regions([("chr1", 0, 4), ("chr2", 2, 5)])
    # covered positions only: chr1[0,4) -> flat 0..3, chr2[2,5) -> flat 12..14
    assert view.sizes["position"] == 7
    assert view["flat_pos"].values.tolist() == [0, 1, 2, 3, 12, 13, 14]
    assert view["region"].values.tolist() == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("chunks", [None, 5])
def test_reduce_mean_matches_numpy(chunks):
    ds = make_ds(chunks)
    intervals = [("chr1", 0, 4), ("chr2", 2, 5)]
    got = ds.pbz.regions(intervals).pbz.reduce("mean")
    if chunks:
        got = got.compute()
    for r, (contig, s, e) in enumerate(intervals):
        base = flat(contig, s) - s
        seg = ds["coverage"].values[base + s : base + e]
        assert got["coverage"].sel(region=r).values.tolist() == pytest.approx(seg.mean(axis=0).tolist())


def _numpy_top1(ds, intervals, keys, descending):
    """Reference: per interval, per sample, lexsort-take winner row."""
    n = len(intervals)
    out = {v: np.empty((n, 2)) for v in ("fire_coverage", "coverage")}
    for r, (contig, s, e) in enumerate(intervals):
        base = flat(contig, s) - s
        lo, hi = base + s, base + e
        for j in range(2):
            cols = [ds[k].values[lo:hi, j] for k in keys]
            # lexsort: last key is primary; negate for descending
            order = np.lexsort([(-c if d else c) for c, d in zip(reversed(cols), reversed(descending))])
            win = order[0]
            for v in out:
                out[v][r, j] = ds[v].values[lo + win, j]
    return out


@pytest.mark.parametrize("chunks", [None, 5])
def test_top_single_key(chunks):
    ds = make_ds(chunks)
    intervals = [("chr1", 0, 6), ("chr2", 0, 8)]
    rows = ds.pbz.regions(intervals).pbz.top(1, by="fire_coverage", descending=True).compute() \
        if chunks else ds.pbz.regions(intervals).pbz.top(1, by="fire_coverage", descending=True)
    ref = _numpy_top1(ds, intervals, ["fire_coverage"], [True])
    assert rows["fire_coverage"].values == pytest.approx(ref["fire_coverage"])
    assert rows["coverage"].values == pytest.approx(ref["coverage"])   # coupled: coverage from the same base


def test_top_tiebreak():
    # force ties on the primary key so the secondary key decides
    ds = make_ds()
    ds["fire_coverage"][:] = 3.0                       # all tied on primary
    intervals = [("chr1", 0, 6)]
    rows = ds.pbz.regions(intervals).pbz.top(
        1, by=["fire_coverage", "coverage"], descending=[True, True]
    )
    ref = _numpy_top1(ds, intervals, ["fire_coverage", "coverage"], [True, True])
    assert rows["coverage"].values == pytest.approx(ref["coverage"])
    assert rows["fire_coverage"].values == pytest.approx(ref["fire_coverage"])


def test_top_requires_region_view():
    with pytest.raises(ValueError, match="region view"):
        make_ds().pbz.top(1, by="coverage")


def test_dataframe_interval_input_matches_tuples():
    import pandas as pd

    ds = make_ds()
    tuples = [("chr1", 0, 4), ("chr2", 2, 5)]
    df = pd.DataFrame({"#chrom": ["chr1", "chr2"], "start": [0, 2], "end": [4, 5]})
    a = ds.pbz.regions(tuples).pbz.reduce("mean")
    b = ds.pbz.regions(df).pbz.reduce("mean")
    xr.testing.assert_allclose(a, b)
