"""Genome functions on whole-genome xarray objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pbzarr
from pbzarr import Reference

from _helpers import write_pbz_store

REFERENCE = {"chr1": 6, "chr2": 4}
INTERVALS = ["chr1:0-3", "chr2:1-4"]
VALUES = np.arange(20, dtype=np.int32).reshape(10, 2)
EXPECTED = [VALUES[0:3].mean(axis=0), VALUES[7:10].mean(axis=0)]


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "genome.pbz"
    write_pbz_store(
        path, REFERENCE, "depth", VALUES, dims=("position", "sample"),
        chunks=(4, 2), labels={"sample": ["a", "b"]},
    )
    return path


def test_output_type_is_input_type(store: Path) -> None:
    with pbzarr.open(store) as handle:
        ds = handle.to_xarray().assign(status=("sample", ["case", "control"]))
        array = pbzarr.reduce_regions(ds["depth"], INTERVALS, "mean")
        dataset = pbzarr.reduce_regions(ds, INTERVALS, "mean")
    assert isinstance(array, xr.DataArray)
    assert array.name == "depth"
    np.testing.assert_allclose(array.values, EXPECTED)
    assert isinstance(dataset, xr.Dataset)
    np.testing.assert_allclose(dataset["depth"].values, EXPECTED)
    assert dataset["status"].identical(ds["status"])
    assert "position" in ds["depth"].dims


def test_position_in_any_order_gives_the_same_values(store: Path) -> None:
    with pbzarr.open(store) as handle:
        depth = handle["depth"]
        first = pbzarr.reduce_regions(depth, INTERVALS, "mean")
        last = pbzarr.reduce_regions(depth.transpose("sample", "position"), INTERVALS, "mean")
    np.testing.assert_allclose(last.values, first.values)


def test_sliced_input_is_refused(store: Path) -> None:
    with pbzarr.open(store) as handle:
        part = handle["depth"].isel(position=slice(0, 4))
        with pytest.raises(ValueError, match=r"region\(\)"):
            pbzarr.reduce_regions(part, INTERVALS, "mean")


def test_plain_xarray_input_takes_reference(store: Path) -> None:
    plain = xr.open_zarr(store, consolidated=False)
    with pytest.raises(ValueError, match="pass reference="):
        pbzarr.reduce_regions(plain, INTERVALS, "mean")
    reference = Reference(list(REFERENCE), list(REFERENCE.values()))
    result = pbzarr.reduce_regions(plain, INTERVALS, "mean", reference=reference)
    np.testing.assert_allclose(result["depth"].values, EXPECTED)
    assert Reference.from_xarray(result) is reference


def test_position_coordinates_are_dropped(store: Path) -> None:
    with pbzarr.open(store) as handle:
        ds = handle.to_xarray().assign_coords(pos=("position", np.arange(10)))
        result = pbzarr.reduce_regions(ds["depth"], INTERVALS, "mean")
    np.testing.assert_allclose(result.values, EXPECTED)
    assert "pos" not in result.coords
