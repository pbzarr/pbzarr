"""Open, select, reduce, and write a format 0.5 store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import pbzarr
from pbzarr import PbzError, Reference, RegionQuery
from pbzarr._coord import NAME, ReferenceIndex, attach, same_reference

from _helpers import write_pbz_store

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
        dim="sample",
        chunk_size=1000,
        column_chunk_size=1,
        shard_size=4000,
        shard_column_size=1,
    )
    return path


def test_open_selects_a_region_through_reference_rows(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        assert isinstance(ds, xr.Dataset)
        reference = Reference.from_xarray(ds)
        assert dict(reference) == {"chr1": LENGTH}
        rows = reference.rows("chr1:100-110")
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


def test_store_without_coordinates_attribute_opens_by_name(tmp_path: Path) -> None:
    path = tmp_path / "plain.pbz"
    values = np.arange(12, dtype=np.int16).reshape(6, 2)
    write_pbz_store(
        path, {"chr1": 4, "chr2": 2}, "depth", values, dims=("position", "sample"),
        chunks=(3, 2), labels={"sample": ["a", "b"]}, coordinates_attr=False,
    )
    assert "coordinates" not in zarr.open_group(path, mode="r").attrs
    with pbzarr.open(path, chunks=None) as ds:
        assert dict(Reference.from_xarray(ds)) == {"chr1": 4, "chr2": 2}
        assert list(ds.data_vars) == ["depth"]
        assert {"contig_id", "contig_length", "sample"} <= set(ds.coords)
        np.testing.assert_array_equal(ds["depth"].values, values)


def test_tracks_filter_keeps_the_reference(cohort: Path) -> None:
    with pbzarr.open(cohort, tracks=["depth"]) as ds:
        assert list(ds.data_vars) == ["depth"]
        assert {"contig_id", "contig_length", "sample"} <= set(ds.coords)


def test_reduce_regions_matches_hand_values(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        result = pbzarr.reduce_regions(ds, ["chr1:0-50", "chr1:100-110"], "mean")
    expected = [
        [_expected(s, np.arange(0, 50)).mean() for s in (0, 1)],
        [_expected(s, np.arange(100, 110)).mean() for s in (0, 1)],
    ]
    np.testing.assert_allclose(result["depth"].values, expected)
    assert result["region_contig"].values.tolist() == ["chr1", "chr1"]


def test_derived_write_round_trips(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        callable_count = (ds["depth"] >= 120).sum("sample")
        pbzarr.write(cohort, {"callable_count": callable_count}).close()
    with pbzarr.open(cohort) as ds:
        positions = np.arange(LENGTH)
        expected = (_expected(0, positions) >= 120).astype(int) + (
            _expected(1, positions) >= 120
        ).astype(int)
        np.testing.assert_array_equal(ds["callable_count"].values, expected)
        assert ds.attrs["pbz:version"] == "0.5"


def test_write_reads_each_source_block_once_for_many_variables(cohort: Path) -> None:
    reads = []

    def count(block):
        reads.append(block.shape)
        return block

    with pbzarr.open(cohort) as ds:
        depth = ds["depth"]
        counted = depth.copy(
            data=depth.data.map_blocks(count, dtype=depth.dtype, meta=np.empty((0, 0), depth.dtype))
        )
        thresholds = [1, 10, 50, 100, 120]
        variables = {"mean": counted.mean("sample")}
        variables |= {f"over_{t}": (counted >= t).mean("sample") for t in thresholds}
        pbzarr.write(ds, variables).close()
        assert len(reads) == counted.data.npartitions

    with pbzarr.open(cohort) as ds:
        values = ds["depth"].values
        np.testing.assert_allclose(ds["mean"].values, values.mean(1))
        for t in thresholds:
            np.testing.assert_allclose(ds[f"over_{t}"].values, (values >= t).mean(1))


def test_write_returns_the_reopened_store_and_leaves_the_input(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        first = ds["depth"].isel(sample=0)
        with pbzarr.write(cohort, {"first": first}) as by_path:
            np.testing.assert_array_equal(by_path["first"].values, first.values)
        with pbzarr.write(ds, {"second": first + 1}) as by_dataset:
            assert {"first", "second"} <= set(by_dataset.data_vars)
            assert isinstance(by_dataset.xindexes[NAME], ReferenceIndex)
        assert list(ds.data_vars) == ["depth"]


def test_write_refusals_name_the_problem(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        short = ds["depth"].isel(position=slice(0, 10), sample=0)
        with pytest.raises(ValueError, match="has 10 rows on 'position'"):
            pbzarr.write(ds, {"short": short})
        with pytest.raises(ValueError, match="genomic dimension name"):
            pbzarr.write(ds, {"position": ds["depth"].isel(sample=0)})
        pbzarr.write(cohort, {"first": ds["depth"].isel(sample=0)}).close()
        # The checks read the store of now, not the snapshot that ds holds.
        with pytest.raises(ValueError, match="'first' already exists"):
            pbzarr.write(ds, {"first": ds["depth"].isel(sample=0)})


def test_legacy_store_is_refused(tmp_path: Path, d4_fixture: Path) -> None:
    group = zarr.open_group(tmp_path / "old.pbz", mode="w", zarr_format=3)
    group.attrs["zarr_conventions"] = [
        {"uuid": "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c", "name": "perbase"}
    ]
    with pytest.raises(PbzError, match="This is a format 0.4 store"):
        pbzarr.open(tmp_path / "old.pbz")
    with pytest.raises(PbzError, match="This is a format 0.4 store"):
        pbzarr.import_d4(tmp_path / "old.pbz", "depth", d4_fixture)


def test_version_rule(cohort: Path) -> None:
    group = zarr.open_group(cohort, mode="r+")
    group.attrs["pbz:future"] = "ignored"
    with pbzarr.open(cohort) as ds:
        first = ds["depth"].isel(sample=0)
    group.attrs["pbz:version"] = "9.9"
    message = "Unsupported format version 9.9. This pbzarr reads 0.5."
    with pytest.raises(PbzError, match=message):
        pbzarr.open(cohort)
    with pytest.raises(PbzError, match=message):
        pbzarr.write(cohort, {"first": first})
    del group.attrs["pbz:version"]
    with pytest.raises(PbzError, match="not a valid PBZ store"):
        pbzarr.open(cohort)


def test_xarray_reads_a_python_written_store(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        pbzarr.write(ds, {"mean_depth": ds["depth"].astype("float32").mean("sample")}).close()
    native = xr.open_zarr(cohort, consolidated=False)
    assert native["mean_depth"].dims == ("position",)
    assert "_zarrs" not in native["mean_depth"].attrs
    native.close()


def test_boolean_track_regions_keep_logical_values(cohort: Path) -> None:
    # Xarray stores a bool track as int8 with dtype="bool"; region reads must
    # decode it the same way plain reads do.
    with pbzarr.open(cohort) as ds:
        pbzarr.write(ds, {"high": ds["depth"].isel(sample=1) >= 110}).close()
    intervals = ["chr1:95-115", "chr1:0-10"]
    with pbzarr.open(cohort) as ds:
        direct = [pbzarr.region(ds["high"], query).values.max() for query in intervals]
        reduced = pbzarr.reduce_regions(ds["high"], intervals, "max").values
    assert reduced.dtype == bool
    assert reduced.tolist() == [True, False]
    np.testing.assert_array_equal(reduced, direct)


def test_summit_reduces_an_opened_dataset(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        result = pbzarr.reduce_regions(ds, ["chr1:95-115"], "summit", by=("depth",))
    assert result["summit_position"].values.tolist() == [[110, 110]]
    assert result["depth"].values.tolist() == [[12, 112]]


def test_failed_write_leaves_no_track(cohort: Path) -> None:
    import dask.array as da

    def fail(block):
        raise RuntimeError("compute failed")

    with pbzarr.open(cohort) as ds:
        bad = ds["depth"].isel(sample=0).copy(
            data=da.map_blocks(fail, ds["depth"].isel(sample=0).data, dtype="int32")
        )
        with pytest.raises(RuntimeError, match="compute failed"):
            pbzarr.write(ds, {"bad": bad})
    assert "bad" not in zarr.open_group(cohort, mode="r")
    with pbzarr.open(cohort) as ds:
        assert "bad" not in ds.data_vars


def test_opened_arrays_carry_the_reference_coordinate(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        reference = Reference.from_xarray(ds)
        for obj in (ds["depth"], ds.data_vars["depth"], ds):
            assert isinstance(obj.xindexes[NAME], ReferenceIndex)
            assert same_reference(Reference.from_xarray(obj), reference)


def test_write_checks_the_reference_and_drops_it(cohort: Path) -> None:
    other = Reference(["chrX"], [LENGTH])
    with pbzarr.open(cohort) as ds:
        first = ds["depth"].isel(sample=0)
        pbzarr.write(ds, {"indexed": first, "plain": first.drop_indexes(NAME)}).close()
        foreign = attach(first.drop_vars(NAME), other)
        with pytest.raises(xr.AlignmentError, match="another reference"):
            pbzarr.write(ds, {"foreign": foreign})
        with pytest.raises(xr.AlignmentError, match="another reference"):
            pbzarr.write(ds, {"foreign": foreign.drop_indexes(NAME)})
    group = zarr.open_group(cohort, mode="r")
    assert {"indexed", "plain"} <= set(group.array_keys())
    assert NAME not in group


def test_string_metadata_variable_round_trips(cohort: Path) -> None:
    status = xr.DataArray(
        np.array(["case", "control"]), dims=("sample",), coords={"sample": ["s0", "s1"]}
    )
    pbzarr.write(cohort, {"status": status}).close()
    with pbzarr.open(cohort, chunks=None) as ds:
        assert ds["status"].values.tolist() == ["case", "control"]


def test_write_saves_a_non_dimension_coordinate(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        mean = ds["depth"].astype("float32").mean("position")
        grouped = mean.assign_coords(sample_group=("sample", ["g1", "g2"]))
        pbzarr.write(ds, {"mean_depth": grouped}).close()
    stored = zarr.open_group(cohort, mode="r")["sample_group"]
    assert stored.metadata.dimension_names == ("sample",)
    assert stored[:].tolist() == ["g1", "g2"]


def test_write_refuses_misplaced_coordinates(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        pos = ds["depth"].isel(sample=0).assign_coords(pos=("position", np.arange(LENGTH)))
        with pytest.raises(ValueError, match="'pos' lies on a genomic dimension"):
            pbzarr.write(ds, {"with_pos": pos})
        wrong = xr.DataArray(
            np.zeros((2, 3)),
            dims=("sample", "cohort"),
            coords={"sample": ["s0", "s1"], "cohort": ("sample", ["a", "b"])},
        )
        with pytest.raises(ValueError, match="label coordinate 'cohort' lies on"):
            pbzarr.write(ds, {"wrong": wrong})


def test_write_drops_0d_coordinates(cohort: Path) -> None:
    with pbzarr.open(cohort) as ds:
        depth = ds["depth"].astype("float64")
        pbzarr.write(ds, {"q50": depth.quantile(0.5, "position")}).close()
        pbzarr.write(ds, {"q90": depth.quantile(0.9, "position")}).close()
    with pbzarr.open(cohort) as ds:
        assert {"q50", "q90"} <= set(ds.data_vars)
        assert "quantile" not in ds.variables
        for name in ds.data_vars:
            assert "quantile" not in ds[name].coords


def test_version_is_a_string() -> None:
    assert isinstance(pbzarr.__version__, str)
    assert pbzarr.__version__


def test_numeric_float_fill_value_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "fill.pbz"
    write_pbz_store(
        path, {"chr1": 10}, "x", np.zeros(10, np.float32), dims=("position",), chunks=(10,)
    )
    zarr.open_array(path / "x", mode="r+").attrs["_FillValue"] = -1.0
    with pytest.raises(PbzError, match="x: floating _FillValue -1.0 is a JSON number"):
        pbzarr.open(path)
