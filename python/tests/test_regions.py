from __future__ import annotations

import hashlib

import cloudpickle
import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

import pbzarr
from pbzarr._region import RegionQuery
from pbzarr._regions import (
    RegionLayout,
    _normalize_one,
    _resolve_region,
    _resolve_regions,
    regions_dataset,
)
from pbzarr._xarray import _compose_track_datasets


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
    if hasattr(store, "reads"):
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


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        (
            "chr2:1-4",
            ([2], [1], [4], [5], [8], [0, 3], [0]),
        ),
        (
            ("chr2", 1, 4),
            ([2], [1], [4], [5], [8], [0, 3], [0]),
        ),
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
            ([0, 2], [0, 1], [2, 4], [0, 5], [2, 8], [0, 2, 5], [1, 0]),
        ),
        (
            (
                np.asarray(["chr2", "chr1"]),
                np.asarray([1, 0]),
                np.asarray([4, 2]),
            ),
            ([0, 2], [0, 1], [2, 4], [0, 5], [2, 8], [0, 2, 5], [1, 0]),
        ),
        (
            iter([("chr2", 1, 4), ("chr1", 0, 2)]),
            ([0, 2], [0, 1], [2, 4], [0, 5], [2, 8], [0, 2, 5], [1, 0]),
        ),
        (
            (("chr2", 1, 4), ("chr1", 0, 2), ("chr2", 4, 6)),
            (
                [0, 2, 2],
                [0, 1, 4],
                [2, 4, 6],
                [0, 5, 8],
                [2, 8, 10],
                [0, 2, 5, 7],
                [1, 0, 2],
            ),
        ),
    ],
    ids=["query", "scalar-tuple", "frame", "columns", "rows", "tuple-rows"],
)
def test_resolve_regions_normalizes_supported_inputs_into_stable_packed_layout(
    intervals, expected
):
    layout = _resolve_regions(_track_dataset(), intervals)
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
    assert all(
        not array.flags.writeable
        for array in (
            layout.contig_ids,
            layout.starts,
            layout.stops,
            layout.flat_starts,
            layout.flat_stops,
            layout.packed_offsets,
            layout.input_index,
        )
    )


@pytest.mark.parametrize(
    "frame",
    [
        _Frame(["start", "stop"], {"start": [0], "stop": [1]}),
        _Frame(
            ["contig", "chrom", "start", "stop"],
            {"contig": ["chr1"], "chrom": ["chr1"], "start": [0], "stop": [1]},
        ),
        _Frame(
            ["contig", "start", "stop", "end"],
            {"contig": ["chr1"], "start": [0], "stop": [1], "end": [1]},
        ),
        _Frame(["contig", "stop"], {"contig": ["chr1"], "stop": [1]}),
    ],
    ids=["missing-contig", "ambiguous-contig", "ambiguous-stop", "missing-start"],
)
def test_resolve_regions_rejects_frames_without_one_required_alias(frame):
    with pytest.raises(ValueError):
        _resolve_regions(_track_dataset(), frame)


@pytest.mark.parametrize(
    "intervals",
    [
        ("chr1", True, 1),
        (np.asarray(["chr1"]), np.asarray([0.5]), np.asarray([1])),
        (np.asarray(["chr1"]), np.asarray([np.nan]), np.asarray([1])),
        (
            np.asarray(["chr1"]),
            np.asarray([0], dtype=object),
            np.asarray([1]),
        ),
        ("chr1", 0, 2**63),
        (np.asarray(["chr1"]), np.asarray([[0]]), np.asarray([1])),
        (np.asarray(["chr1"]), np.asarray([0, 1]), np.asarray([1])),
        [],
        ("missing", 0, 1),
        ("chr1", 1, 1),
        ("chr1", 3, 2),
        ("chr1", -1, 1),
        ("chr1", 0, 5),
        (
            np.asarray(["chr1", "chr1"]),
            np.asarray([0, 1]),
            np.asarray([3, 2]),
        ),
    ],
    ids=[
        "boolean",
        "float",
        "nonfinite-nan",
        "object-coercion",
        "signed-64-overflow",
        "non-1d",
        "unequal-columns",
        "empty",
        "unknown-contig",
        "empty-interval",
        "reversed-interval",
        "negative-start",
        "out-of-bounds",
        "overlap",
    ],
)
def test_resolve_regions_rejects_invalid_bulk_interval_inputs(intervals):
    with pytest.raises((TypeError, ValueError, KeyError)):
        _resolve_regions(_track_dataset(), intervals)


def test_resolve_regions_allows_adjacent_unsorted_intervals():
    layout = _resolve_regions(
        _track_dataset(),
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


def test_public_regions_returns_stable_packed_dataset_without_indexes():
    store, expected = _store_track()
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.pbz.regions(
        [("chr2", 1, 4), ("chr1", 1, 3)]
    )

    assert list(result.data_vars) == ["values"]
    assert result["values"].dims == ("position",)
    assert result["values"].chunks == ((5,),)
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
    assert result.attrs == {
        "pbz:representation": "packed-regions",
        "pbz:parent_genome_checksum": dataset.attrs[
            "perbase:genome_checksum"
        ],
        "pbz:coordinates": "0-based-half-open",
    }
    assert not _value_reads(store)

    np.testing.assert_array_equal(
        result["values"].compute(),
        np.concatenate((expected[1:3], expected[5:8])),
    )
    reads = _value_reads(store)
    assert any("values/c/0" in key for key in reads)
    assert any("values/c/1" in key for key in reads)
    assert not any("values/c/2" in key for key in reads)


def test_reduce_regions_falls_back_after_an_xarray_value_transformation():
    store, _ = _store_track()
    dataset = pbzarr.open(store)
    transformed = dataset.assign(values=dataset["values"] + 1)

    reduced = transformed.pbz.reduce_regions(
        [("chr1", 0, 2), ("chr2", 0, 3)],
        "mean",
    )

    np.testing.assert_array_equal(reduced.compute()["values"], [1.5, 6.0])


def test_reduce_regions_selects_summits_through_the_public_accessor():
    dataset = _track_dataset()

    reduced = dataset.pbz.reduce_regions(
        [("chr1", 0, 4), ("chr2", 0, 3)],
        "summit",
        by=("values",),
    ).compute()

    np.testing.assert_array_equal(reduced["values"], [[6, 7], [12, 13]])
    np.testing.assert_array_equal(reduced["summit_position"], [[3, 3], [2, 2]])


def test_reduce_regions_builds_compact_self_contained_pbz_tasks(tmp_path):
    path = tmp_path / "track.pbz"
    _, expected = _store_track(store=path)
    dataset = pbzarr.open(path)

    reduced = dataset.pbz.reduce_regions(
        [("chr1", 0, 2), ("chr2", 0, 3)],
        "mean",
    )

    graph = reduced["values"].data.__dask_graph__().to_dict()
    tasks = [task for task in graph.values() if hasattr(task, "func")]
    assert len(tasks) == 1
    assert not tasks[0].dependencies
    assert len(cloudpickle.dumps(graph)) < 32_000
    np.testing.assert_array_equal(
        reduced.compute()["values"],
        [expected[0:2].mean(), expected[4:7].mean()],
    )


def test_regions_dataset_keeps_one_source_chunk_in_one_batch():
    source = da.from_array(
        np.arange(10, dtype=np.int16), chunks=(10,), name="shared-source"
    )
    track = _track_dataset().drop_vars(["values", "sample"]).assign(
        values=("position", source)
    )

    result = regions_dataset(
        track,
        [("chr1", 0, 1), ("chr1", 2, 3)],
        target_bytes=100,
        max_source_blocks=1,
    )

    dependencies = result["values"].data.dask.get_all_dependencies()
    source_key = ("shared-source", 0)
    gather_dependents = [
        key for key, required in dependencies.items() if source_key in required
    ]
    assert len(gather_dependents) == 1
    assert result["values"].chunks == ((2,),)
    np.testing.assert_array_equal(result["values"].compute(), [0, 2])


def test_public_regions_preserves_two_dimensional_non_position_chunks():
    store, expected = _store_track(two_dimensional=True, sharded=True)
    dataset = pbzarr.open(store)
    store.reads.clear()

    result = dataset.pbz.regions(
        [("chr2", 1, 3), ("chr1", 1, 4)]
    )

    assert result["values"].dims == ("position", "context")
    assert result["values"].chunks == ((5,), (2, 1))
    assert result["context"].values.tolist() == ["CG", "CHG", "CHH"]
    assert not result.xindexes
    assert not _value_reads(store)
    np.testing.assert_array_equal(
        result["values"].compute(),
        np.concatenate((expected[1:4], expected[5:7]), axis=0),
    )


def test_public_regions_consumes_unequal_in_memory_dask_chunks():
    dataset = _track_dataset().chunk(
        {"position": (3, 4, 3), "sample": (1, 1)}
    )

    result = dataset.pbz.regions(
        [("chr2", 0, 5), ("chr1", 1, 2)]
    )

    assert result["values"].chunks == ((6,), (1, 1))
    np.testing.assert_array_equal(
        result["values"].compute(),
        np.concatenate(
            (
                np.arange(20).reshape(10, 2)[1:2],
                np.arange(20).reshape(10, 2)[4:9],
            ),
            axis=0,
        ),
    )


def test_public_regions_chunks_none_reads_only_selected_pieces_eagerly():
    store, expected = _store_track()
    dataset = pbzarr.open(store, chunks=None)
    store.reads.clear()

    result = dataset.pbz.regions([("chr2", 1, 3)])

    assert result["values"].chunks is None
    assert isinstance(result["values"].data, np.ndarray)
    np.testing.assert_array_equal(result["values"], expected[5:7])
    reads = _value_reads(store)
    assert any("values/c/1" in key for key in reads)
    assert not any("values/c/0" in key for key in reads)
    assert not any("values/c/2" in key for key in reads)


def test_public_regions_accepts_exact_composed_dataset_and_preserves_close():
    depth = _track_dataset().chunk(
        {"position": (3, 4, 3), "sample": (1, 1)}
    )
    mask = xr.Dataset(
        {
            "values": (
                "position",
                da.from_array(
                    np.arange(10, dtype=np.int16), chunks=(6, 4)
                ),
            )
        },
        coords=xr.Coordinates(
            {
                "contigs": depth["contigs"].variable,
                "offsets": depth["offsets"].variable,
            },
            indexes={},
        ),
        attrs=depth.attrs,
    )
    composed = _compose_track_datasets({"depth": depth, "mask": mask})
    closed = []
    composed.set_close(lambda: closed.append(True))

    result = composed.pbz.regions([("chr2", 1, 3)])

    assert list(result.data_vars) == ["depth", "mask"]
    assert result["depth"].dims == ("position", "sample")
    assert result["mask"].dims == ("position",)
    assert result["depth"].chunks == ((2,), (1, 1))
    assert result["mask"].chunks == ((2,),)
    assert result["sample"].values.tolist() == ["A", "B"]
    np.testing.assert_array_equal(
        result["depth"], np.arange(20).reshape(10, 2)[5:7]
    )
    np.testing.assert_array_equal(result["mask"], np.arange(10)[5:7])
    result.close()
    assert closed == [True]


def test_datatree_regions_uses_only_selected_children():
    depth = _track_dataset()
    mask = xr.Dataset(
        {"values": ("position", np.arange(10, dtype=np.int16))},
        coords=xr.Coordinates(
            {
                "contigs": depth["contigs"].variable,
                "offsets": depth["offsets"].variable,
            },
            indexes={},
        ),
        attrs=depth.attrs,
    )
    tree = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "/depth": depth, "/mask": mask}
    )

    result = tree.pbz.regions([("chr1", 1, 3)], tracks=["mask"])

    assert list(result.data_vars) == ["mask"]
    np.testing.assert_array_equal(result["mask"], [1, 2])


def test_public_regions_rejects_packed_or_position_dependent_composed_input():
    packed = _track_dataset().assign_attrs(
        {"pbz:representation": "packed-regions"}
    )
    composed = _compose_track_datasets({"depth": _track_dataset()}).assign_coords(
        genomic_position=("position", np.arange(10))
    )

    with pytest.raises(pbzarr.PbzError):
        packed.pbz.regions([("chr1", 0, 2)])
    with pytest.raises(pbzarr.PbzError):
        composed.pbz.regions([("chr1", 0, 2)])


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


def test_public_namespace_is_the_xarray_api():
    public_names = {
        name for name in vars(pbzarr) if not name.startswith("_")
    }
    expected = {
        "PbzError",
        "open",
        "RegionQuery",
        "parse_region",
        "create_store",
        "import_d4",
        "import_bigwig",
        "import_bed",
        "import_bed_multi",
        "import_bam",
        "stack",
    }
    assert set(pbzarr.__all__) == public_names == expected
