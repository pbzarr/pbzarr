from __future__ import annotations

from dataclasses import fields
import hashlib

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

import pbzarr
from pbzarr._region import RegionQuery, parse_region
from pbzarr._regions import _normalize_one, _resolve_region


class _CountingStore(MemoryStore):
    def __init__(self, store_dict=None, *, read_only=False, reads=None):
        super().__init__(store_dict=store_dict, read_only=read_only)
        self.reads = [] if reads is None else reads

    async def get(self, key, prototype, byte_range=None):
        self.reads.append(key)
        return await super().get(key, prototype, byte_range)

    def with_read_only(self, read_only=False):
        return type(self)(
            store_dict=self._store_dict,
            read_only=read_only,
            reads=self.reads,
        )


def _store_track(*, two_dimensional=False, sharded=False):
    store = _CountingStore()
    group = zarr.open_group(store, mode="w", zarr_format=3)
    contigs = np.asarray(["chr1", "empty", "chr2"])
    offsets = np.asarray([0, 4, 4, 10], dtype=np.int64)
    dimensions = ("position", "context") if two_dimensional else ("position",)
    shape = (10, 3) if two_dimensional else (10,)
    chunks = (4, 1) if two_dimensional else (4,)
    shards = (8, 2) if sharded else None
    values = group.create_array(
        "values",
        shape=shape,
        chunks=chunks,
        shards=shards,
        dtype="int16",
        fill_value=-1,
        dimension_names=dimensions,
    )
    expected = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    values[:] = expected
    contig_array = group.create_array(
        "contigs",
        shape=(3,),
        chunks=(3,),
        dtype="str",
        dimension_names=("contig",),
    )
    contig_array[:] = contigs
    group.create_array(
        "offsets",
        data=offsets,
        chunks=(4,),
        dimension_names=("contig_boundary",),
    )
    if two_dimensional:
        labels = group.create_array(
            "context",
            shape=(3,),
            chunks=(3,),
            dtype="str",
            dimension_names=("context",),
        )
        labels[:] = np.asarray(["CG", "CHG", "CHH"])
    payload = "chr1\t4\nchr2\t6\nempty\t0\n"
    group.attrs.update(
        {
            "zarr_conventions": [{"uuid": "test", "name": "perbase"}],
            "perbase:kind": "track",
            "perbase:version": "0.4",
            "perbase:coordinates": "0-based-half-open",
            "perbase:genome_checksum": "md5:"
            + hashlib.md5(payload.encode()).hexdigest(),
        }
    )
    store.reads.clear()
    return store, expected


def _value_reads(store):
    return [key for key in store.reads if "/values/c/" in f"/{key}"]


def _track_dataset() -> xr.Dataset:
    contigs = np.asarray(["chr1", "empty", "chr2"])
    offsets = np.asarray([0, 4, 4, 10], dtype=np.int64)
    payload = "chr1\t4\nchr2\t6\nempty\t0\n"
    coordinates = xr.Coordinates(
        {
            "contigs": ("contig", contigs),
            "offsets": ("contig_boundary", offsets),
            "sample": ("sample", ["A", "B"]),
        },
        indexes={},
    )
    return xr.Dataset(
        {"values": (("position", "sample"), np.arange(20).reshape(10, 2))},
        coords=coordinates,
        attrs={
            "zarr_conventions": [{"uuid": "test", "name": "perbase"}],
            "perbase:kind": "track",
            "perbase:version": "0.4",
            "perbase:coordinates": "0-based-half-open",
            "perbase:genome_checksum": "md5:"
            + hashlib.md5(payload.encode()).hexdigest(),
        },
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (RegionQuery("chr2", 1, 4), RegionQuery("chr2", 1, 4)),
        ("chr2:1-4", RegionQuery("chr2", 1, 4)),
        (("chr2", 1, 4), RegionQuery("chr2", 1, 4)),
        (("chr2", None, None), RegionQuery("chr2", None, None)),
        ("chr2:1", RegionQuery("chr2", 1, None)),
    ],
)
def test_normalize_one_accepts_only_canonical_single_query_forms(query, expected):
    assert _normalize_one(query) == expected


def test_region_query_uses_stop_as_its_only_endpoint_name():
    parsed = parse_region("chr2:1-4")

    assert parsed == RegionQuery("chr2", 1, 4)
    assert [field.name for field in fields(parsed)] == ["contig", "start", "stop"]
    assert not hasattr(parsed, "end")


@pytest.mark.parametrize(
    "query",
    [
        RegionQuery("", 0, 1),
        RegionQuery("chr1", True, 2),
        RegionQuery("chr1", 0.5, 2),
        RegionQuery("chr1", np.nan, 2),
        RegionQuery("chr1", 0, 2**63),
        ("chr1", object(), 2),
        ["chr1", 0, 2],
    ],
)
def test_normalize_one_rejects_noncanonical_or_non_integral_queries(query):
    with pytest.raises((TypeError, ValueError)):
        _normalize_one(query)


@pytest.mark.parametrize(
    "query",
    [
        RegionQuery("missing", 0, 1),
        RegionQuery("chr1", -1, 2),
        RegionQuery("chr1", 2, 2),
        RegionQuery("chr1", 3, 2),
        RegionQuery("chr1", 0, 5),
        RegionQuery("empty", None, None),
    ],
)
def test_resolve_region_rejects_invalid_contig_local_bounds(query):
    with pytest.raises((KeyError, ValueError)):
        _resolve_region(_track_dataset(), query)


def test_resolve_region_fills_bounds_and_returns_one_exact_flat_slice():
    normalized, flat_slice = _resolve_region(
        _track_dataset(), ("chr2", None, None)
    )

    assert normalized == RegionQuery("chr2", 0, 6)
    assert flat_slice == slice(4, 10)


def test_resolve_region_offsets_a_bounded_query_on_the_flat_axis():
    normalized, flat_slice = _resolve_region(_track_dataset(), "chr2:1-4")

    assert normalized == RegionQuery("chr2", 1, 4)
    assert flat_slice == slice(5, 8)


def test_public_region_slice_is_lazy_and_reads_only_intersecting_regular_chunks():
    store, expected = _store_track()
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.pbz.region("chr2:1-5")

    assert result.dims == ("position",)
    assert result.dtype == expected.dtype
    assert result.chunks == ((3, 1),)
    assert "position" not in result.coords
    assert not result.xindexes
    assert result.coords["region_contig"].item() == "chr2"
    assert result.coords["region_start"].item() == 1
    assert result.coords["region_stop"].item() == 5
    assert not _value_reads(store)

    assert result.compute().values.tolist() == expected[5:9].tolist()
    reads = _value_reads(store)
    assert any("values/c/1" in key for key in reads)
    assert any("values/c/2" in key for key in reads)
    assert not any("values/c/0" in key for key in reads)


def test_public_region_slice_keeps_chunks_none_backend_lazy():
    store, expected = _store_track()
    dataset = pbzarr.open(store, chunks=None)
    store.reads.clear()

    result = dataset.pbz.region(("chr1", 1, 3))

    assert result.chunks is None
    assert not _value_reads(store)
    assert result.values.tolist() == expected[1:3].tolist()
    assert _value_reads(store)


def test_public_region_slice_preserves_sharded_two_dimensional_values():
    store, expected = _store_track(two_dimensional=True, sharded=True)
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.pbz.region("chr1:1-4")

    assert result.dims == ("position", "context")
    assert result.dtype == expected.dtype
    assert result.chunks == ((3,), (2, 1))
    assert not _value_reads(store)
    np.testing.assert_array_equal(result.compute(), expected[1:4, :])


def test_public_region_column_selects_the_declared_generic_dimension():
    store, expected = _store_track(two_dimensional=True, sharded=True)
    dataset = pbzarr.open(store)

    result = dataset.pbz.region("chr1:1-4", column="CHG")

    assert result.dims == ("position",)
    np.testing.assert_array_equal(result.compute(), expected[1:4, 1])
    with pytest.raises(KeyError):
        dataset.pbz.region("chr1:1-4", column="missing")


def test_public_region_slice_consumes_unequal_in_memory_dask_chunks():
    dataset = _track_dataset().chunk({"position": (3, 4, 3), "sample": 1})

    result = dataset.pbz.region("chr2:0-5")

    assert result.chunks == ((3, 2), (1, 1))
    np.testing.assert_array_equal(
        result.compute(), np.arange(20).reshape(10, 2)[4:9]
    )


@pytest.mark.parametrize(
    "dataset",
    [
        _track_dataset().assign(extra=_track_dataset()["values"]),
        _track_dataset().assign_attrs({"pbz:representation": "packed-regions"}),
        _track_dataset().transpose("sample", "position", ...),
    ],
    ids=["composed", "packed", "transposed"],
)
def test_public_region_slice_rejects_non_track_dataset_shapes(dataset):
    with pytest.raises(pbzarr.PbzError):
        dataset.pbz.region("chr1:0-2")


def test_region_query_and_parser_are_public():
    assert pbzarr.RegionQuery is RegionQuery
    assert pbzarr.parse_region("chr1:0-2") == RegionQuery("chr1", 0, 2)
