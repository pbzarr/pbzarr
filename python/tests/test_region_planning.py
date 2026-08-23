from __future__ import annotations

import functools
import hashlib
import pickle

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

import pbzarr
from pbzarr._regions import (
    RegionLayout,
    _gather_dask,
    _gather_block,
    _plan_variable_regions,
)


def _layout(
    flat_starts: list[int], flat_stops: list[int]
) -> RegionLayout:
    starts = np.asarray(flat_starts, dtype=np.int64)
    stops = np.asarray(flat_stops, dtype=np.int64)
    lengths = stops - starts
    packed_offsets = np.empty(len(starts) + 1, dtype=np.int64)
    packed_offsets[0] = 0
    np.cumsum(lengths, out=packed_offsets[1:])
    arrays = (
        np.zeros(len(starts), dtype=np.int64),
        starts,
        stops,
        starts.copy(),
        stops.copy(),
        packed_offsets,
        np.arange(len(starts), dtype=np.int64),
    )
    for array in arrays:
        array.setflags(write=False)
    return RegionLayout(*arrays)


def _values(*, chunks) -> xr.DataArray:
    data = da.arange(50, dtype=np.int16, chunks=10).reshape((10, 5))
    data = data.rechunk(chunks)
    return xr.DataArray(data, dims=("position", "context"))


def test_plan_variable_regions_emits_exact_read_only_pieces_for_unequal_chunks():
    values = _values(chunks=((3, 4, 3), (2, 3)))
    layout = _layout([2], [9])

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=1_000,
        max_source_blocks=16,
    )

    assert values.dims == ("position", "context")
    assert values.chunks == ((3, 4, 3), (2, 3))
    np.testing.assert_array_equal(batch_regions, [0, 1])
    np.testing.assert_array_equal(batch_pieces, [0, 3])
    assert pieces.tolist() == [(0, 2, 3), (1, 0, 4), (2, 0, 2)]
    assert pieces.dtype.names == ("source_block", "source_start", "source_stop")
    assert all(
        array.dtype == np.dtype("int32")
        for array in (batch_regions, batch_pieces)
    )
    assert all(
        pieces.dtype.fields[name][0] == np.dtype("int32")
        for name in pieces.dtype.names
    )
    assert all(
        not array.flags.writeable
        for array in (batch_regions, batch_pieces, pieces)
    )


def test_plan_variable_regions_keeps_one_source_chunk_in_one_batch():
    values = xr.DataArray(
        da.arange(20, dtype=np.int16, chunks=(20,)), dims=("position",)
    )
    layout = _layout([0, 2, 4, 6], [1, 3, 5, 7])

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=4,
        max_source_blocks=2,
    )

    np.testing.assert_array_equal(batch_regions, [0, 4])
    np.testing.assert_array_equal(batch_pieces, [0, 4])
    assert pieces["source_block"].tolist() == [0, 0, 0, 0]


def test_ten_thousand_regions_in_four_source_chunks_make_one_gather_task():
    source_size = 40_000
    source_chunk_size = 10_000
    starts = np.arange(0, source_size, 4, dtype=np.int64)
    n_regions = starts.size
    layout = _layout(starts, starts + 1)
    values = xr.DataArray(
        da.arange(
            source_size,
            chunks=(source_chunk_size,),
            dtype=np.int64,
        ),
        dims=("position",),
    )

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=1_000_000,
        max_source_blocks=4,
    )

    assert len(batch_regions) - 1 == 1
    assert len(batch_pieces) - 1 == 1
    assert np.unique(pieces["source_block"]).tolist() == [0, 1, 2, 3]

    gathered = _gather_dask(
        values,
        layout,
        batch_regions,
        batch_pieces,
        pieces,
    )

    graph = gathered.__dask_graph__().to_dict()
    gather_tasks = [
        task
        for task in graph.values()
        if getattr(task, "func", None) is _gather_block
    ]

    assert len(gather_tasks) == 1
    gather_task = gather_tasks[0]
    assert len(gather_task.dependencies) == 4
    assert gathered.chunks == ((n_regions,),)
    np.testing.assert_array_equal(gathered.compute(), starts)


def test_plan_variable_regions_keeps_an_oversized_source_component_whole():
    values = xr.DataArray(
        da.arange(20, dtype=np.int16, chunks=(4, 4, 4, 4, 4)),
        dims=("position",),
    )
    layout = _layout([1, 18], [18, 20])

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=10,
        max_source_blocks=2,
    )

    np.testing.assert_array_equal(batch_regions, [0, 2])
    np.testing.assert_array_equal(batch_pieces, [0, 6])
    assert pieces.tolist() == [
        (0, 1, 4),
        (1, 0, 4),
        (2, 0, 4),
        (3, 0, 4),
        (4, 0, 2),
        (4, 2, 4),
    ]


def test_plan_variable_regions_splits_before_a_source_component():
    values = xr.DataArray(
        da.arange(12, dtype=np.int16, chunks=(4, 4, 4)),
        dims=("position",),
    )
    layout = _layout(
        np.array([0, 4, 9]),
        np.array([1, 9, 10]),
    )

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=13,
        max_source_blocks=10,
    )

    np.testing.assert_array_equal(batch_regions, [0, 1, 3])
    np.testing.assert_array_equal(batch_pieces, [0, 1, 4])
    assert pieces.tolist() == [
        (0, 0, 1),
        (1, 0, 4),
        (2, 0, 1),
        (2, 1, 2),
    ]


def test_plan_variable_regions_rejects_non_position_first_or_dependent_coordinates():
    transposed = _values(chunks=((3, 4, 3), (2, 3))).transpose(
        "context", "position"
    )
    positioned = _values(chunks=((3, 4, 3), (2, 3))).assign_coords(
        genomic_position=("position", np.arange(10))
    )

    with pytest.raises(pbzarr.PbzError):
        _plan_variable_regions(
            transposed,
            _layout([0], [1]),
            target_bytes=100,
            max_source_blocks=16,
        )
    with pytest.raises(pbzarr.PbzError):
        _plan_variable_regions(
            positioned,
            _layout([0], [1]),
            target_bytes=100,
            max_source_blocks=16,
        )


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


def _eager_track() -> tuple[_CountingStore, xr.Dataset]:
    store = _CountingStore()
    group = zarr.open_group(store, mode="w", zarr_format=3)
    group.create_array(
        "values",
        data=np.arange(10, dtype=np.int16),
        chunks=(4,),
        dimension_names=("position",),
    )
    group.create_array(
        "contigs",
        data=np.asarray(["chr1"]),
        chunks=(1,),
        dimension_names=("contig",),
    )
    group.create_array(
        "offsets",
        data=np.asarray([0, 10], dtype=np.int64),
        chunks=(2,),
        dimension_names=("contig_boundary",),
    )
    payload = "chr1\t10\n"
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
    dataset = pbzarr.open(store, chunks=None)
    store.reads.clear()
    return store, dataset


def test_plan_variable_regions_inspects_chunks_none_without_reading_values():
    store, dataset = _eager_track()

    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        dataset["values"],
        _layout([3], [7]),
        target_bytes=100,
        max_source_blocks=16,
    )

    np.testing.assert_array_equal(batch_regions, [0, 1])
    np.testing.assert_array_equal(batch_pieces, [0, 1])
    assert pieces.tolist() == [(0, 3, 7)]
    assert not [key for key in store.reads if "/values/c/" in f"/{key}"]


def test_two_million_regions_plan_in_typed_linear_storage_without_values():
    n_regions = 2_000_000
    stride = 1_000_000
    starts = np.arange(n_regions, dtype=np.int64) * stride
    layout = _layout(starts, starts + 1)
    logical_positions = n_regions * stride
    variables = (
        xr.DataArray(
            da.empty(
                (logical_positions,),
                chunks=(1_000_000_000,),
                dtype=np.int16,
            ),
            dims=("position",),
        ),
        xr.DataArray(
            da.empty(
                (logical_positions, 76),
                chunks=(1_000_000_000, 19),
                dtype=np.float32,
            ),
            dims=("position", "sample"),
        ),
    )

    layout_arrays = tuple(vars(layout).values())
    assert layout.n_regions == n_regions
    assert layout.total_positions == n_regions
    assert all(
        np.issubdtype(array.dtype, np.signedinteger)
        and not array.flags.writeable
        for array in layout_arrays
    )
    assert sum(array.nbytes for array in layout_arrays) == (
        7 * n_regions + 1
    ) * np.dtype(np.int64).itemsize

    expected_batches = 125
    for values in variables:
        batch_regions, batch_pieces, pieces = _plan_variable_regions(
            values,
            layout,
            target_bytes=128 * 1024**2,
            max_source_blocks=16,
        )

        assert values.size >= 2_000_000_000_000
        assert len(batch_regions) == expected_batches + 1
        assert len(batch_pieces) == expected_batches + 1
        assert len(pieces) == n_regions
        assert all(
            np.issubdtype(array.dtype, np.signedinteger)
            and not array.flags.writeable
            for array in (batch_regions, batch_pieces)
        )
        assert all(
            np.issubdtype(pieces.dtype.fields[name][0], np.signedinteger)
            for name in pieces.dtype.names
        )
        assert not pieces.dtype.hasobject
        assert not pieces.flags.writeable
        itemsize = batch_regions.dtype.itemsize
        assert sum(
            array.nbytes for array in (batch_regions, batch_pieces, pieces)
        ) == itemsize * (
            2 * (expected_batches + 1) + 3 * n_regions
        )
        del batch_regions, batch_pieces, pieces


def _packed_graph_metrics(n_regions: int):
    starts = np.arange(n_regions, dtype=np.int64) * 2
    layout = _layout(starts, starts + 1)
    values = xr.DataArray(
        da.empty(
            (n_regions * 2,),
            chunks=(64,),
            dtype=np.int16,
            name=f"bounded-source-{n_regions}",
        ),
        dims=("position",),
    )
    values.encoding["source"] = "/sentinel/do-not-open.pbz"
    batch_regions, batch_pieces, pieces = _plan_variable_regions(
        values,
        layout,
        target_bytes=10_000,
        max_source_blocks=8,
    )
    gathered = _gather_dask(
        values,
        layout,
        batch_regions,
        batch_pieces,
        pieces,
    )
    graph = gathered.__dask_graph__()
    payload = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    callables = []
    for task in graph.to_dict().values():
        function = getattr(task, "func", None)
        while isinstance(function, functools.partial):
            function = function.func
        if callable(function):
            callables.append(
                f"{function.__module__}.{function.__qualname__}"
            )
    return (
        len(graph),
        len(payload),
        len(values.chunks[0]),
        len(batch_regions) - 1,
        payload,
        callables,
    )


def test_packed_graph_grows_with_blocks_and_batches_without_reopen_tasks():
    small = _packed_graph_metrics(512)
    large = _packed_graph_metrics(1_024)

    for task_count, _, source_blocks, batches, payload, callables in (
        small,
        large,
    ):
        assert task_count <= 4 * (source_blocks + batches)
        assert b"/sentinel/do-not-open.pbz" not in payload
        assert not any("open" in name.casefold() for name in callables)
    assert large[2] == 2 * small[2]
    assert large[3] == 2 * small[3]
    assert large[0] <= 2 * small[0]
    assert large[1] < 3 * small[1]


def test_gather_block_fuses_cross_boundary_pieces_in_order():
    pieces = np.asarray(
        [(0, 2, 4), (1, 0, 2)],
        dtype=[
            ("source_block", "i4"),
            ("source_start", "i4"),
            ("source_stop", "i4"),
        ],
    )

    gathered = _gather_block(
        (np.arange(4, dtype=np.int16), np.arange(4, 8, dtype=np.int16)),
        pieces,
        (4,),
    )

    np.testing.assert_array_equal(gathered, [2, 3, 4, 5])


def test_contig_ids_accepts_stringdtype_contigs():
    from pbzarr._regions import _contig_ids

    contigs = np.array(["chr1", "chr2", "HLA-A*01:01"], dtype=np.dtypes.StringDType())
    names = np.asarray(["chr2", "HLA-A*01:01", "chr1"])
    ids = _contig_ids(contigs, names)
    assert ids.tolist() == [1, 2, 0]
