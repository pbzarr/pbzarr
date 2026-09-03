from __future__ import annotations

import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

import pbzarr  # noqa: F401
from pbzarr._regions import (
    _reduce_packed_dataset,
    _validate_packed_dataset,
    regions_dataset,
)


def _packed_dataset(*, lazy: bool = False, multiple: bool = False) -> xr.Dataset:
    offsets = np.asarray([0, 2, 5, 9], dtype=np.int64)
    depth = np.asarray(
        [1.0, 3.0, np.nan, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
    )
    matrix = np.column_stack(
        (
            depth,
            [2.0, 4.0, 1.0, 3.0, 5.0, 1.0, np.nan, 7.0, 9.0],
            [6.0, 6.0, 3.0, 3.0, 3.0, 8.0, 8.0, 8.0, 8.0],
        )
    )
    if lazy:
        depth = da.from_array(depth, chunks=(5, 4))
        matrix = da.from_array(matrix, chunks=((2, 7), (1, 2)))
    data_vars = {"depth": ("position", depth)}
    if multiple:
        data_vars["signal"] = (("position", "context"), matrix)
    return xr.Dataset(
        data_vars=data_vars,
        coords=xr.Coordinates(
            {
                "offsets": ("region_boundary", offsets),
                "region_contig": ("region", ["chr1", "chr1", "chr2"]),
                "region_start": ("region", [0, 4, 2]),
                "region_stop": ("region", [2, 7, 6]),
                "region_input_index": ("region", [1, 2, 0]),
                "region_storage_index": ("region", [0, 1, 2]),
                **(
                    {"context": ("context", ["CG", "CHG", "CHH"])}
                    if multiple
                    else {}
                ),
            },
            indexes={},
        ),
        attrs={
            "pbz:representation": "packed-regions",
            "pbz:parent_genome_checksum": "md5:test",
            "pbz:coordinates": "0-based-half-open",
        },
    )


def test_reduce_mean_returns_an_ordinary_region_dataset() -> None:
    packed = _packed_dataset()
    packed["depth"].encoding["_FillValue"] = -1.0

    actual = _reduce_packed_dataset(packed, "mean")

    expected = xr.Dataset(
        data_vars={"depth": ("region", [2.0, 6.0, 5.0])},
        coords=xr.Coordinates(
            {
                "region_contig": ("region", ["chr1", "chr1", "chr2"]),
                "region_start": ("region", [0, 4, 2]),
                "region_stop": ("region", [2, 7, 6]),
                "region_input_index": ("region", [1, 2, 0]),
                "region_storage_index": ("region", [0, 1, 2]),
            },
            indexes={},
        ),
    )
    xr.testing.assert_identical(actual, expected)

    assert actual["depth"].encoding == {}
    assert not actual.xindexes


def test_summit_selects_one_position_per_region_and_column() -> None:
    fire = np.asarray(
        [
            [5, 1],
            [5, 3],
            [4, 3],
            [0, 4],
            [2, 4],
            [2, 1],
        ],
        dtype=np.int32,
    )
    coverage = np.asarray(
        [
            [7, 9],
            [8, 5],
            [100, 6],
            [9, 8],
            [3, 7],
            [3, 10],
        ],
        dtype=np.int32,
    )
    marker = np.asarray(
        [
            [100, 101],
            [110, 111],
            [120, 121],
            [130, 131],
            [140, 141],
            [150, 151],
        ],
        dtype=np.int32,
    )
    packed = xr.Dataset(
        data_vars={
            "fire_coverage": (("position", "sample"), fire),
            "coverage": (("position", "sample"), coverage),
            "marker": (("position", "sample"), marker),
        },
        coords=xr.Coordinates(
            {
                "offsets": ("region_boundary", [0, 3, 6]),
                "region_contig": ("region", ["chr1", "chr2"]),
                "region_start": ("region", [10, 20]),
                "region_stop": ("region", [13, 23]),
                "region_input_index": ("region", [0, 1]),
                "region_storage_index": ("region", [0, 1]),
                "sample": ("sample", ["s1", "s2"]),
            },
            indexes={},
        ),
        attrs={
            "pbz:representation": "packed-regions",
            "pbz:parent_genome_checksum": "md5:test",
            "pbz:coordinates": "0-based-half-open",
        },
    )

    actual = _reduce_packed_dataset(
        packed,
        "summit",
        by=("fire_coverage", "coverage"),
    )

    expected = xr.Dataset(
        data_vars={
            "fire_coverage": (
                ("region", "sample"),
                [[5, 3], [2, 4]],
            ),
            "coverage": (
                ("region", "sample"),
                [[8, 6], [3, 8]],
            ),
            "marker": (
                ("region", "sample"),
                [[110, 121], [140, 131]],
            ),
        },
        coords=xr.Coordinates(
            {
                "region_contig": ("region", ["chr1", "chr2"]),
                "region_start": ("region", [10, 20]),
                "region_stop": ("region", [13, 23]),
                "region_input_index": ("region", [0, 1]),
                "region_storage_index": ("region", [0, 1]),
                "sample": ("sample", ["s1", "s2"]),
                "summit_position": (
                    ("region", "sample"),
                    [[11, 12], [21, 20]],
                ),
            },
            indexes={},
        ),
    )
    xr.testing.assert_identical(actual, expected)

    lazy_input = packed.assign(
        {
            name: values.chunk({"position": 3, "sample": 1})
            for name, values in packed.data_vars.items()
        }
    )
    lazy = _reduce_packed_dataset(
        lazy_input,
        "summit",
        by=("fire_coverage", "coverage"),
    )
    assert lazy.chunks is not None
    xr.testing.assert_identical(lazy.compute(), expected)


@pytest.mark.parametrize(
    ("reducer", "kwargs"),
    [
        ("mean", {}),
        ("count", {}),
        ("std", {"ddof": 1}),
        ("median", {}),
        ("quantile", {"q": 0.5}),
        ("quantile", {"q": [0.25, 0.75]}),
    ],
    ids=[
        "mean",
        "count",
        "std-ddof-one",
        "median",
        "scalar-quantile",
        "vector-quantile",
    ],
)
def test_reduce_matches_named_xarray_groupby_for_scalar_and_2d_variables(
    reducer: str, kwargs: dict
) -> None:
    packed = _packed_dataset(multiple=True)
    _assert_matches_groupby(packed, reducer, kwargs)


@pytest.mark.parametrize(
    ("reducer", "kwargs"),
    [
        ("median", {}),
        ("quantile", {"q": [0.25, 0.75]}),
    ],
    ids=["median", "vector-quantile"],
)
def test_reduce_uses_complete_region_tasks_for_dask_variables_with_different_chunks(
    reducer: str, kwargs: dict
) -> None:
    packed = _packed_dataset(lazy=True, multiple=True)
    assert packed["depth"].chunks[0] != packed["signal"].chunks[0]

    _assert_matches_groupby(packed, reducer, kwargs)


def _assert_matches_groupby(
    packed: xr.Dataset, reducer: str, kwargs: dict
) -> None:
    labels = xr.DataArray(
        np.repeat(np.arange(3), np.diff(packed["offsets"].data)),
        dims=("position",),
        name="region",
    )

    actual = _reduce_packed_dataset(packed, reducer, **kwargs).compute()

    for name, values in packed.data_vars.items():
        eager = xr.DataArray(
            values.compute().data,
            dims=values.dims,
            attrs=values.attrs,
        )
        method = getattr(eager.groupby(labels), reducer)
        expected = method(dim="position", **kwargs)
        xr.testing.assert_allclose(
            xr.DataArray(actual[name].variable),
            xr.DataArray(expected.variable),
        )
    assert actual["context"].data.tolist() == ["CG", "CHG", "CHH"]
    if reducer == "quantile" and np.ndim(kwargs["q"]) == 0:
        assert actual["signal"].dims == ("region", "context")
        assert actual["quantile"].dims == ()
    if reducer == "quantile" and np.ndim(kwargs["q"]) == 1:
        assert actual["signal"].dims == ("region", "context", "quantile")
        assert actual["quantile"].dims == ("quantile",)


def test_validate_packed_dataset_returns_a_read_only_offsets_copy() -> None:
    packed = _packed_dataset()

    offsets = _validate_packed_dataset(packed)

    assert not np.shares_memory(offsets, packed["offsets"].data)
    assert not offsets.flags.writeable
    assert packed["offsets"].data.flags.writeable


def test_reduce_rejects_non_packed_and_corrupted_packed_datasets() -> None:
    packed = _packed_dataset(multiple=True)
    corruptions = [
        packed.assign_attrs({"pbz:representation": "track"}),
        packed.drop_vars("region_contig"),
        packed.assign_coords(position=np.arange(9)),
        packed.assign_coords(offsets=("region_boundary", [0, 2, 5, 8])),
        packed.assign_coords(offsets=("region_boundary", [0, 2, 2, 9])),
        packed.assign_coords(region_stop=("region", [2, 8, 6])),
        packed.assign_coords(region_input_index=("region", [0, 0, 2])),
        packed.assign_coords(region_storage_index=("region", [1, 0, 2])),
        packed.isel(position=slice(1, None)),
        packed.isel(region=slice(1, None)),
        packed.sortby("region_input_index"),
        packed.transpose("context", "position", ...),
    ]

    for corrupted in corruptions:
        with pytest.raises(ValueError):
            _reduce_packed_dataset(corrupted, "mean")


def test_reduce_has_an_exact_reducer_allowlist_and_owns_the_position_dim() -> None:
    packed = _packed_dataset()

    with pytest.raises(ValueError, match="unsupported reducer"):
        _reduce_packed_dataset(packed, "prod")
    with pytest.raises(TypeError, match="controls dim='position'"):
        _reduce_packed_dataset(packed, "mean", dim="position")


def test_reduce_passes_xarray_kwargs_and_preserves_reducer_attrs_policy() -> None:
    packed = _packed_dataset().assign(
        depth=_packed_dataset()["depth"].assign_attrs(units="reads")
    )

    with_nan = _reduce_packed_dataset(
        packed, "mean", skipna=False, keep_attrs=False
    )
    with_attrs = _reduce_packed_dataset(packed, "mean", keep_attrs=True)

    assert np.isnan(with_nan["depth"].data[1])
    assert with_nan["depth"].attrs == {}
    assert with_attrs["depth"].attrs == {"units": "reads"}


def _record_block(reads: list[int], index: int, values: np.ndarray):
    reads.append(index)
    return values


def test_reduce_is_lazy_and_culls_to_a_bounded_cross_chunk_region() -> None:
    reads: list[int] = []
    source_blocks = []
    source_values = np.arange(9, dtype=np.float64)
    for index, (start, stop) in enumerate(((0, 3), (3, 6), (6, 9))):
        task = delayed(_record_block, pure=False)(
            reads, index, source_values[start:stop]
        )
        source_blocks.append(
            da.from_delayed(task, shape=(stop - start,), dtype=np.float64)
        )
    track = xr.Dataset(
        {"values": ("position", da.concatenate(source_blocks))},
        coords=xr.Coordinates(
            {
                "contigs": ("contig", ["chr1"]),
                "offsets": ("contig_boundary", [0, 9]),
            },
            indexes={},
        ),
        attrs={
            "zarr_conventions": [{"uuid": "test", "name": "perbase"}],
            "perbase:kind": "track",
            "perbase:version": "0.4",
            "perbase:coordinates": "0-based-half-open",
            "perbase:genome_checksum": "md5:8579e0e056d92d53101d9b70ba03f797",
        },
    )
    closed = []
    track.set_close(lambda: closed.append(True))
    packed = regions_dataset(
        track,
        [("chr1", 1, 5), ("chr1", 6, 9)],
        target_bytes=8,
        max_source_blocks=16,
    )

    reduced = _reduce_packed_dataset(packed, "mean")

    assert reads == []
    assert reduced["values"].chunks == ((1, 1),)
    assert reduced["values"].isel(region=slice(0, 1)).compute().item() == 2.5
    assert set(reads) == {0, 1}
    reduced.close()
    assert closed == [True]


def test_non_position_xarray_operations_compose_on_either_side_of_reduce() -> None:
    packed = _packed_dataset(multiple=True)

    position_then_column = _reduce_packed_dataset(packed, "mean")["signal"].max(
        "context"
    )
    column_then_position = _reduce_packed_dataset(
        packed.max("context"), "mean"
    )["signal"]

    np.testing.assert_allclose(position_then_column, [6.0, 6.0, 8.0])
    np.testing.assert_allclose(column_then_position, [6.0, 5.0, 8.25])
    assert not position_then_column.equals(column_then_position)
