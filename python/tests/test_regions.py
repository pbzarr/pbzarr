from __future__ import annotations

import cloudpickle
import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

import pbzarr
from pbzarr._region import RegionQuery
from pbzarr._regions import (
    RegionLayout,
    _normalize_one,
    _resolve_one,
    _resolve_regions,
    reduce_regions_dataset,
)
from _helpers import write_pbz_store

CONTIGS = {"chr1": 4, "empty": 0, "chr2": 6}
REFERENCE = pbzarr.Reference(list(CONTIGS), list(CONTIGS.values()))


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


def _store_track(*, two_dimensional=False, sharded=False, store=None):
    store = _CountingStore() if store is None else store
    shape = (10, 3) if two_dimensional else (10,)
    expected = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    write_pbz_store(
        store,
        CONTIGS,
        "depth",
        expected,
        dims=("position", "context") if two_dimensional else ("position",),
        chunks=(4, 1) if two_dimensional else (4,),
        shards=(8, 2) if sharded else None,
        labels={"context": ["CG", "CHG", "CHH"]} if two_dimensional else None,
    )
    if hasattr(store, "reads"):
        store.reads.clear()
    return store, expected


def _value_reads(store):
    return [key for key in store.reads if "/depth/c/" in f"/{key}"]


def _track_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"depth": (("position", "sample"), np.arange(20).reshape(10, 2))},
        coords={"sample": ["A", "B"]},
    )


class _Columns:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self):
        return self._values


class _Frame:
    def __init__(self, columns, values):
        self.columns = columns
        self._values = values

    def __getitem__(self, name):
        return self._values[name]


_TWO_REGIONS = ([0, 2], [0, 1], [2, 4], [0, 5], [2, 8], [0, 2, 5], [1, 0])


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ("chr2:1-4", ([2], [1], [4], [5], [8], [0, 3], [0])),
        (("chr2", 1, 4), ([2], [1], [4], [5], [8], [0, 3], [0])),
        (
            _Frame(
                ["#chrom", "start", "end", "ignored"],
                {
                    "#chrom": _Columns(["chr2", "chr1"]),
                    "start": _Columns([1, 0]),
                    "end": _Columns([4, 2]),
                    "ignored": _Columns(["a", "b"]),
                },
            ),
            _TWO_REGIONS,
        ),
        (
            (np.asarray(["chr2", "chr1"]), np.asarray([1, 0]), np.asarray([4, 2])),
            _TWO_REGIONS,
        ),
        (iter([("chr2", 1, 4), ("chr1", 0, 2)]), _TWO_REGIONS),
        (["chr2:1-4", "chr1:0-2"], _TWO_REGIONS),
        ([RegionQuery("chr2", 1, 4), RegionQuery("chr1", None, 2)], _TWO_REGIONS),
    ],
    ids=["query", "scalar-tuple", "frame", "columns", "rows", "strings", "queries"],
)
def test_resolve_regions_normalizes_supported_inputs_into_stable_packed_layout(
    intervals, expected
):
    layout = _resolve_regions(_track_dataset(), REFERENCE, intervals)
    (
        contig_ids,
        starts,
        stops,
        flat_starts,
        flat_stops,
        packed_offsets,
        input_index,
    ) = expected

    assert isinstance(layout, RegionLayout)
    np.testing.assert_array_equal(layout.contig_ids, contig_ids)
    np.testing.assert_array_equal(layout.starts, starts)
    np.testing.assert_array_equal(layout.stops, stops)
    np.testing.assert_array_equal(layout.flat_starts, flat_starts)
    np.testing.assert_array_equal(layout.flat_stops, flat_stops)
    np.testing.assert_array_equal(layout.packed_offsets, packed_offsets)
    np.testing.assert_array_equal(layout.input_index, input_index)
    assert layout.n_regions == len(contig_ids)
    assert layout.total_positions == packed_offsets[-1]


@pytest.mark.parametrize(
    "intervals",
    [
        [("missing", 0, 1)],
        [("chr1", 3, 2)],
        [("chr1", -1, 1)],
        [("chr1", 0, 5)],
        ["chr1:0-5"],
        [("chr1", 0, 3), ("chr1", 1, 2)],
    ],
    ids=[
        "unknown-contig",
        "reversed-interval",
        "negative-start",
        "out-of-range",
        "out-of-range-string",
        "overlap",
    ],
)
def test_resolve_regions_rejects_out_of_range_and_overlapping_intervals(intervals):
    with pytest.raises((ValueError, KeyError)):
        _resolve_regions(_track_dataset(), REFERENCE, intervals)


def test_resolve_regions_allows_adjacent_unsorted_intervals():
    layout = _resolve_regions(
        _track_dataset(),
        REFERENCE,
        (
            np.asarray(["chr2", "chr1", "chr1"]),
            np.asarray([1, 2, 0]),
            np.asarray([3, 4, 2]),
        ),
    )

    np.testing.assert_array_equal(layout.contig_ids, [0, 0, 2])
    np.testing.assert_array_equal(layout.starts, [0, 2, 1])
    np.testing.assert_array_equal(layout.stops, [2, 4, 3])
    np.testing.assert_array_equal(layout.input_index, [2, 1, 0])


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


@pytest.mark.parametrize(
    ("query", "resolved", "flat"),
    [
        ("chr2", RegionQuery("chr2", 0, 6), slice(4, 10)),
        ("chr2:1-4", RegionQuery("chr2", 1, 4), slice(5, 8)),
    ],
)
def test_resolve_one_fills_bounds_and_offsets_onto_the_flat_axis(
    query, resolved, flat
):
    assert _resolve_one(REFERENCE, _normalize_one(query)) == (resolved, flat)


def test_single_region_is_lazy_and_reads_only_intersecting_chunks():
    store, expected = _store_track()
    dataset = pbzarr.open(store)
    assert not _value_reads(store)
    store.reads.clear()

    result = dataset.regions("chr2:1-5")

    assert result["depth"].dims == ("position",)
    assert result["depth"].dtype == expected.dtype
    assert result["region_contig"].values.tolist() == ["chr2"]
    assert result["region_start"].values.tolist() == [1]
    assert result["region_stop"].values.tolist() == [5]
    assert not _value_reads(store)

    np.testing.assert_array_equal(result["depth"].compute(), expected[5:9])
    reads = _value_reads(store)
    assert any("depth/c/1" in key for key in reads)
    assert any("depth/c/2" in key for key in reads)
    assert not any("depth/c/0" in key for key in reads)


@pytest.mark.parametrize(
    "intervals",
    [[("chr2", 1, 4), ("chr1", 1, 3)], ["chr2:1-4", "chr1:1-3"]],
    ids=["rows", "strings"],
)
def test_regions_returns_stable_packed_dataset_without_indexes(intervals):
    store, expected = _store_track()
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.regions(intervals)

    assert list(result.data_vars) == ["depth"]
    assert result["depth"].dims == ("position",)
    assert result["depth"].chunks == ((5,),)
    assert dict(result.sizes) == {
        "position": 5,
        "region_boundary": 3,
        "region": 2,
    }
    assert not result.xindexes
    np.testing.assert_array_equal(result["offsets"], [0, 2, 5])
    np.testing.assert_array_equal(result["region_contig"], ["chr1", "chr2"])
    np.testing.assert_array_equal(result["region_start"], [1, 1])
    np.testing.assert_array_equal(result["region_stop"], [3, 4])
    np.testing.assert_array_equal(result["region_input_index"], [1, 0])
    np.testing.assert_array_equal(result["region_storage_index"], [0, 1])
    assert not _value_reads(store)

    np.testing.assert_array_equal(
        result["depth"].compute(),
        np.concatenate((expected[1:3], expected[5:8])),
    )
    reads = _value_reads(store)
    assert any("depth/c/0" in key for key in reads)
    assert any("depth/c/1" in key for key in reads)
    assert not any("depth/c/2" in key for key in reads)


def test_regions_preserves_sharded_two_dimensional_non_position_chunks():
    store, expected = _store_track(two_dimensional=True, sharded=True)
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.regions([("chr2", 1, 3), ("chr1", 1, 4)])

    assert result["depth"].dims == ("position", "context")
    assert result["depth"].chunks == ((5,), (2, 1))
    assert result["context"].values.tolist() == ["CG", "CHG", "CHH"]
    assert not result.xindexes
    assert not _value_reads(store)
    np.testing.assert_array_equal(
        result["depth"].compute(),
        np.concatenate((expected[1:4], expected[5:7]), axis=0),
    )


def test_regions_chunks_none_reads_only_selected_pieces_eagerly():
    store, expected = _store_track()
    dataset = pbzarr.open(store, chunks=None)
    store.reads.clear()

    result = dataset.regions([("chr2", 1, 3)])

    assert result["depth"].chunks is None
    assert isinstance(result["depth"].data, np.ndarray)
    np.testing.assert_array_equal(result["depth"], expected[5:7])
    reads = _value_reads(store)
    assert any("depth/c/1" in key for key in reads)
    assert not any("depth/c/0" in key for key in reads)
    assert not any("depth/c/2" in key for key in reads)


def test_reduce_regions_falls_back_after_an_xarray_value_transformation(tmp_path):
    path = tmp_path / "track.pbz"
    _store_track(store=path)
    dataset = pbzarr.open(path)
    transformed = dataset.to_xarray().assign(depth=dataset["depth"] + 1)

    reduced = reduce_regions_dataset(
        transformed,
        dataset.reference,
        [("chr1", 0, 2), ("chr2", 0, 3)],
        "mean",
    )

    np.testing.assert_array_equal(reduced.compute()["depth"], [1.5, 6.0])


def test_reduce_regions_builds_compact_self_contained_pbz_tasks(tmp_path):
    path = tmp_path / "track.pbz"
    _, expected = _store_track(store=path)
    dataset = pbzarr.open(path)

    reduced = dataset.reduce_regions(
        [("chr1", 0, 2), ("chr2", 0, 3)],
        "mean",
    )

    graph = reduced["depth"].data.__dask_graph__().to_dict()
    tasks = [task for task in graph.values() if hasattr(task, "func")]
    assert len(tasks) == 1
    assert not tasks[0].dependencies
    assert len(cloudpickle.dumps(graph)) < 32_000
    np.testing.assert_array_equal(
        reduced.compute()["depth"],
        [expected[0:2].mean(), expected[4:7].mean()],
    )
