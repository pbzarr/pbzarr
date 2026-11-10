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
    with pbzarr.open(store) as opened:
        ds = opened.assign(status=("sample", ["case", "control"]))
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
    with pbzarr.open(store) as ds:
        depth = ds["depth"]
        first = pbzarr.reduce_regions(depth, INTERVALS, "mean")
        last = pbzarr.reduce_regions(depth.transpose("sample", "position"), INTERVALS, "mean")
    np.testing.assert_allclose(last.values, first.values)


def test_sliced_input_is_refused(store: Path) -> None:
    with pbzarr.open(store) as ds:
        part = ds["depth"].isel(position=slice(0, 4))
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
    with pbzarr.open(store) as opened:
        ds = opened.assign_coords(pos=("position", np.arange(10)))
        result = pbzarr.reduce_regions(ds["depth"], INTERVALS, "mean")
    np.testing.assert_allclose(result.values, EXPECTED)
    assert "pos" not in result.coords


def test_region_of_a_data_array(store: Path) -> None:
    with pbzarr.open(store) as ds:
        depth = ds["depth"]
        locus = pbzarr.region(depth, "chr2:1-4")
        whole = pbzarr.region(depth, ("chr1", None, None))
    np.testing.assert_array_equal(locus.values, VALUES[7:10])
    np.testing.assert_array_equal(locus["pos"].values, [1, 2, 3])
    assert list(locus["position"].values) == ["chr2:1", "chr2:2", "chr2:3"]
    assert (locus.region_contig.item(), locus.region_start.item(), locus.region_end.item()) == (
        "chr2", 1, 4,
    )
    assert Reference.from_xarray(locus).lengths.tolist() == [6, 4]
    np.testing.assert_array_equal(whole.values, VALUES[0:6])


def test_region_of_a_dataset_slices_every_genomic_dimension(store: Path) -> None:
    with pbzarr.open(store) as opened:
        ds = opened.assign(binned=("bin_2", np.arange(5)))
        view = pbzarr.region(ds, "chr2:1-4")
    np.testing.assert_array_equal(view["depth"].values, VALUES[7:10])
    np.testing.assert_array_equal(view["binned"].values, [3, 4])
    assert view["contig_id"].variable.identical(ds["contig_id"].variable)


def test_a_view_is_refused(store: Path) -> None:
    with pbzarr.open(store) as ds:
        view = pbzarr.region(ds["depth"], "chr1")
        with pytest.raises(ValueError, match=r"region\(\)"):
            pbzarr.region(view, "chr1:0-2")
        with pytest.raises(ValueError, match=r"region\(\)"):
            pbzarr.reduce_regions(view, ["chr1:0-2"], "mean")


def test_a_whole_contig_view_builds_no_per_base_labels() -> None:
    import tracemalloc

    import dask.array as da

    from pbzarr._coord import attach

    reference = Reference(["chr1", "chr2"], [248_956_422, 10])
    genome = attach(
        xr.DataArray(da.zeros(248_956_432, chunks=10_000_000), dims="position"), reference
    )
    tracemalloc.start()
    view = pbzarr.region(genome, "chr1")
    repr(view)
    view.isel(position=slice(-2, None)).compute()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 20_000_000
    assert list(view["position"][-2:].values) == ["chr1:248956420", "chr1:248956421"]
    assert view["pos"][-1].item() == 248_956_421


def test_views_align_by_label_within_one_contig(store: Path) -> None:
    with pbzarr.open(store) as ds:
        depth = ds["depth"]
        a = pbzarr.region(depth, "chr1:0-4")
        b = pbzarr.region(depth, "chr1:2-6")
        other = pbzarr.region(depth, "chr2:0-4")
    total = a + b
    assert list(total["position"].values) == ["chr1:2", "chr1:3"]
    np.testing.assert_array_equal(total.values, 2 * VALUES[2:4])
    np.testing.assert_array_equal(a.sel(position="chr1:3").values, VALUES[3])
    with pytest.raises(xr.AlignmentError):
        a + other
