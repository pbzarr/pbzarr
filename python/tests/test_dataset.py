"""The format 0.5 store handle: open, select, reduce, write."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import pbzarr
from pbzarr import PbzError, RegionQuery

LENGTH = 10_000


def _expected(sample: int, positions: np.ndarray) -> np.ndarray:
    """Committed cohort fixture: bands of 10 positions, 100 * sample + band % 50 + 1."""
    return 100 * sample + (positions // 10) % 50 + 1


@pytest.fixture
def cohort(tmp_path: Path, cohort_fixtures: list[Path]) -> Path:
    path = tmp_path / "cohort.pbz"
    pbzarr.create(path, reference={"chr1": LENGTH}).close()
    pbzarr.import_d4(
        path,
        "depth",
        [(cohort_fixtures[0], "s0"), (cohort_fixtures[1], "s1")],
        column_dim="sample",
        chunk_size=1000,
        column_chunk_size=1,
        shard_size=4000,
        shard_column_size=1,
    )
    return path


def test_open_selects_a_region_through_reference_rows(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        assert dict(ds.reference) == {"chr1": LENGTH}
        rows = ds.reference.rows("chr1:100-110")
        assert rows == slice(100, 110)
        view = ds["depth"].isel(position=rows).sel(sample="s1")
        np.testing.assert_array_equal(view.values, _expected(1, np.arange(100, 110)))
        # Default chunks follow the stored shard, not the inner chunk.
        assert ds["depth"].chunks[0][0] == 4000


def test_reference_maps_binned_rows_and_intervals() -> None:
    reference = pbzarr.Reference(["chr1", "chr2"], [10, 7])
    assert reference.size("bin_4") == 3 + 2
    assert reference.rows("chr2:1-5", dim="bin_4") == slice(3, 5)
    assert reference.interval(2, "bin_4") == RegionQuery("chr1", 8, 10)
    assert reference.rows("chr1:3") == slice(3, 10)
    assert reference.rows("chr2:1-5") == slice(11, 15)
    with pytest.raises(ValueError, match="empty or outside"):
        reference.rows("chr1:5-11")


def test_tracks_filter_keeps_the_reference(cohort: Path) -> None:
    with pbzarr.open(cohort, tracks=["depth"]) as ds:
        assert list(ds.tracks) == ["depth"]
        assert {"contig", "contig_length", "sample"} <= set(ds.coords)


def test_reduce_regions_matches_hand_values(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        result = ds.reduce_regions(["chr1:0-50", "chr1:100-110"], "mean")
    expected = [
        [_expected(s, np.arange(0, 50)).mean() for s in (0, 1)],
        [_expected(s, np.arange(100, 110)).mean() for s in (0, 1)],
    ]
    np.testing.assert_allclose(result["depth"].values, expected)
    assert result["region_contig"].values.tolist() == ["chr1", "chr1"]


def test_derived_write_round_trips(cohort: Path) -> None:
    with pbzarr.open(cohort, mode="r+") as ds:
        callable_count = (ds["depth"] >= 120).sum("sample")
        ds.write({"callable_count": callable_count})
        assert "callable_count" in ds
    with pbzarr.open(cohort) as ds:
        positions = np.arange(LENGTH)
        expected = (_expected(0, positions) >= 120).astype(int) + (
            _expected(1, positions) >= 120
        ).astype(int)
        np.testing.assert_array_equal(ds["callable_count"].values, expected)
        assert ds.attrs["pbz:version"] == "0.5"


def test_write_refusals_name_the_problem(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        with pytest.raises(PbzError, match="mode='r\\+'"):
            ds.write({"x": ds["depth"].isel(sample=0)})
    with pbzarr.open(cohort, mode="r+") as ds:
        short = ds["depth"].isel(position=slice(0, 10), sample=0)
        with pytest.raises(ValueError, match="has 10 rows on 'position'"):
            ds.write({"short": short})
        with pytest.raises(ValueError, match="genomic dimension name"):
            ds.write({"position": ds["depth"].isel(sample=0)})
        with pbzarr.open(cohort, mode="r+") as other:
            other.write({"first": other["depth"].isel(sample=0)})
        with pytest.raises(PbzError, match="store changed since open"):
            ds.write({"second": ds["depth"].isel(sample=0)})


def test_legacy_store_is_refused(tmp_path: Path) -> None:
    group = zarr.open_group(tmp_path / "old.pbz", mode="w", zarr_format=3)
    group.attrs["zarr_conventions"] = [
        {"uuid": "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c", "name": "perbase"}
    ]
    with pytest.raises(PbzError, match="format 0.4"):
        pbzarr.open(tmp_path / "old.pbz")


def test_xarray_reads_a_python_written_store(cohort: Path) -> None:
    with pbzarr.open(cohort, mode="r+") as ds:
        ds.write({"mean_depth": ds["depth"].astype("float32").mean("sample")})
    native = xr.open_zarr(cohort, consolidated=False)
    assert native["mean_depth"].dims == ("position",)
    assert "_zarrs" not in native["mean_depth"].attrs
    native.close()


def test_boolean_track_regions_keep_logical_values(cohort: Path) -> None:
    # Xarray stores a bool track as int8 with dtype="bool"; region reads must
    # decode it the same way plain reads do.
    with pbzarr.open(cohort, mode="r+") as ds:
        ds.write({"high": ds["depth"].isel(sample=1) >= 110})
    with pbzarr.open(cohort) as ds:
        direct = ds["high"].isel(position=ds.reference.rows("chr1:95-115")).values
        packed = ds.regions(["chr1:95-115"])["high"].values
    assert packed.dtype == bool
    np.testing.assert_array_equal(packed, direct)


def test_summit_reduces_through_the_handle(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        result = ds.reduce_regions(["chr1:95-115"], "summit", by=("depth",))
    assert result["summit_position"].values.tolist() == [[110, 110]]
    assert result["depth"].values.tolist() == [[12, 112]]


def test_failed_write_leaves_no_track(cohort: Path) -> None:
    import dask.array as da

    def fail(block):
        raise RuntimeError("compute failed")

    with pbzarr.open(cohort, mode="r+") as ds:
        bad = ds["depth"].isel(sample=0).copy(
            data=da.map_blocks(fail, ds["depth"].isel(sample=0).data, dtype="int32")
        )
        with pytest.raises(RuntimeError, match="compute failed"):
            ds.write({"bad": bad})
    assert "bad" not in zarr.open_group(cohort, mode="r")
    with pbzarr.open(cohort) as ds:
        assert "bad" not in ds
