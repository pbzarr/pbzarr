"""reduce_regions through the public API, against NumPy on the same rows."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest
import xarray as xr
import zarr

import pbzarr

from _helpers import write_pbz_store

REFERENCE = {"chr1": 50, "chr2": 30}
OFFSETS = {"chr1": 0, "chr2": 50}
# Unsorted, overlapping, across chunk edges and on two contigs. The rows
# they cover are less than half of the range from the first to the last.
INTERVALS = [("chr2", 20, 26), ("chr1", 10, 16), ("chr1", 4, 12), ("chr1", 14, 15), ("chr1", 40, 45)]
RNG = np.random.default_rng(7)
DEPTH = RNG.integers(0, 100, size=(80, 3)).astype(np.float32)
DEPTH[RNG.random(DEPTH.shape) < 0.1] = np.nan
DEPTH[14] = [1.0, 2.0, 3.0]
COUNT = RNG.integers(0, 1000, size=80).astype(np.int32)


def _add_track(path: Path, name: str, values: np.ndarray, dims, chunks) -> None:
    group = zarr.open_group(path, mode="r+")
    group.create_array(name, data=values, chunks=chunks, dimension_names=dims)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "regions.pbz"
    write_pbz_store(
        path, REFERENCE, "depth", DEPTH, dims=("position", "sample"),
        chunks=(8, 1), labels={"sample": ["a", "b", "c"]},
    )
    _add_track(path, "count", COUNT, ("position",), (8,))
    return path


def _rows(values: np.ndarray, contig: str, start: int, end: int) -> np.ndarray:
    return values[OFFSETS[contig] + start : OFFSETS[contig] + end]


EXPECTED = {
    "sum": (np.nansum, {}),
    "mean": (np.nanmean, {}),
    "min": (np.nanmin, {}),
    "max": (np.nanmax, {}),
    "count": (lambda rows, axis: np.sum(~np.isnan(rows.astype(float)), axis=axis), {}),
    "std": (lambda rows, axis: np.nanstd(rows, axis=axis, ddof=1), {"ddof": 1}),
    "var": (np.nanvar, {}),
    "median": (np.nanmedian, {}),
    "quantile": (
        lambda rows, axis: np.moveaxis(np.nanquantile(rows, [0.25, 0.9], axis=axis), 0, -1),
        {"q": [0.25, 0.9]},
    ),
}


@pytest.mark.parametrize("kind", ["store", "transformed", "eager"])
@pytest.mark.parametrize("reducer", list(EXPECTED))
def test_reducers_match_numpy(store: Path, reducer: str, kind: str) -> None:
    function, options = EXPECTED[reducer]
    with pbzarr.open(store, **({"chunks": None} if kind == "eager" else {})) as ds:
        if kind == "transformed":
            ds = ds.assign(depth=ds["depth"] * 1, count=ds["count"] + 0)
        result = pbzarr.reduce_regions(ds, INTERVALS, reducer, **options)
        assert (result["depth"].chunks is None) == (kind == "eager")
        result = result.compute()
    for name, values in (("depth", DEPTH), ("count", COUNT)):
        with warnings.catch_warnings():
            # std with ddof=1 of the one-row region is NaN, and NumPy says so.
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.stack([function(_rows(values, *r), axis=0) for r in INTERVALS])
        np.testing.assert_allclose(result[name].values, expected, rtol=1e-5)


def test_results_follow_input_order_without_internal_fields(store: Path) -> None:
    with pbzarr.open(store) as ds:
        result = pbzarr.reduce_regions(ds, INTERVALS, "mean")
        array = pbzarr.reduce_regions(ds["count"], INTERVALS, "max")
    assert result["region_contig"].values.tolist() == [r[0] for r in INTERVALS]
    assert result["region_start"].values.tolist() == [r[1] for r in INTERVALS]
    assert result["region_end"].values.tolist() == [r[2] for r in INTERVALS]
    assert set(result.coords) == {
        "region_contig", "region_start", "region_end", "sample", "contig_id",
        "contig_length", "reference",
    }
    assert result.attrs == {}
    assert result["depth"].dims == ("region", "sample")
    assert array.values.tolist() == [int(_rows(COUNT, *r).max()) for r in INTERVALS]


def test_unknown_keyword_is_refused(store: Path) -> None:
    with pbzarr.open(store) as ds:
        with pytest.raises(TypeError):
            pbzarr.reduce_regions(ds, INTERVALS, "mean", target_bytes=8)
        with pytest.raises(TypeError, match="does not take q="):
            pbzarr.reduce_regions(ds, INTERVALS, "mean", q=0.5)
        with pytest.raises(ValueError, match="unsupported reducer"):
            pbzarr.reduce_regions(ds, INTERVALS, "prod")


def test_a_region_inside_one_chunk_reads_only_that_chunk(store: Path) -> None:
    # Store path: every other chunk of the track is garbage, so a read of it fails.
    chunks = sorted((store / "count" / "c").iterdir(), key=lambda p: int(p.name))
    for chunk in chunks:
        if chunk.name != "2":
            chunk.write_bytes(b"not a chunk")
    with pbzarr.open(store, tracks=["count"]) as ds:
        result = pbzarr.reduce_regions(ds, [("chr1", 17, 23), ("chr1", 16, 18)], "sum")
        assert result["count"].values.tolist() == [COUNT[17:23].sum(), COUNT[16:18].sum()]

    # Dask path: count the source blocks that a transformed track computes.
    reads = []

    def count(block, block_info=None):
        reads.append(block_info[0]["chunk-location"])
        return block

    with pbzarr.open(store, tracks=["depth"]) as ds:
        depth = ds["depth"].copy(data=ds["depth"].data.map_blocks(count, dtype=DEPTH.dtype))
        pbzarr.reduce_regions(depth, [("chr1", 33, 38)], "mean").compute()
    assert sorted(reads) == [(4, 0), (4, 1), (4, 2)]


def test_bin_variables_are_left_out_with_a_warning(store: Path) -> None:
    with pbzarr.open(store) as opened:
        ds = opened.assign(binned=("bin_10", np.arange(8)))
        with pytest.warns(UserWarning, match="'binned'"):
            result = pbzarr.reduce_regions(ds, INTERVALS, "mean")
    assert "binned" not in result
    assert {"depth", "count"} <= set(result.data_vars)


def test_summit_picks_the_row_with_the_highest_keys(tmp_path: Path) -> None:
    fire = np.zeros((30, 2), dtype=np.int32)
    coverage = np.zeros((30, 2), dtype=np.int32)
    fire[10:13] = [[5, 1], [5, 3], [4, 3]]
    coverage[10:13] = [[7, 9], [8, 5], [100, 6]]
    fire[20:23] = [[0, 4], [2, 4], [2, 1]]
    coverage[20:23] = [[9, 8], [3, 7], [3, 10]]
    marker = np.arange(60, dtype=np.int32).reshape(30, 2)
    path = tmp_path / "summit.pbz"
    write_pbz_store(
        path, {"chr1": 30}, "fire", fire, dims=("position", "sample"), chunks=(12, 1),
        labels={"sample": ["s1", "s2"]},
    )
    _add_track(path, "coverage", coverage, ("position", "sample"), (12, 1))
    _add_track(path, "marker", marker, ("position", "sample"), (12, 1))
    for options in ({}, {"chunks": None}):
        with pbzarr.open(path, **options) as ds:
            result = pbzarr.reduce_regions(
                ds, ["chr1:20-23", "chr1:10-13"], "summit", by=("fire", "coverage")
            ).compute()
        assert result["summit_position"].values.tolist() == [[21, 20], [11, 12]]
        assert result["fire"].values.tolist() == [[2, 4], [5, 3]]
        assert result["coverage"].values.tolist() == [[3, 8], [8, 6]]
        assert result["marker"].values.tolist() == [[42, 41], [22, 25]]
