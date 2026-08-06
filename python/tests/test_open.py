from __future__ import annotations

import hashlib
import dask
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.storage import MemoryStore

import pbzarr


class CountingMemoryStore(MemoryStore):
    def __init__(
        self, store_dict=None, *, read_only=False, reads=None, listings=None
    ):
        super().__init__(store_dict=store_dict, read_only=read_only)
        self.reads = [] if reads is None else reads
        self.listings = [] if listings is None else listings

    async def get(self, key, prototype, byte_range=None):
        self.reads.append(key)
        return await super().get(key, prototype, byte_range)

    async def list(self):
        self.listings.append(None)
        async for key in super().list():
            yield key

    async def list_prefix(self, prefix):
        self.listings.append(prefix)
        async for key in super().list_prefix(prefix):
            yield key

    async def list_dir(self, prefix):
        self.listings.append(prefix)
        async for key in super().list_dir(prefix):
            yield key

    def with_read_only(self, read_only=False):
        return type(self)(
            store_dict=self._store_dict,
            read_only=read_only,
            reads=self.reads,
            listings=self.listings,
        )


def _checksum(contigs, offsets):
    records = sorted(
        zip(contigs, np.diff(offsets)), key=lambda item: item[0].encode()
    )
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    return "md5:" + hashlib.md5(payload.encode()).hexdigest()


def _populate_track(
    group,
    *,
    contigs=("chr1", "empty", "chr2"),
    offsets=(0, 4, 4, 10),
    shape=None,
    dims=("position",),
    labels=None,
    label_dims=None,
    chunks=None,
    shards=None,
    dtype="int16",
    fill_value=-7,
    kind="track",
    attrs=None,
    offsets_dtype=np.int64,
    missing=(),
    extra_array=None,
):
    shape = (offsets[-1],) if shape is None else shape
    chunks = tuple(max(1, length) for length in shape) if chunks is None else chunks

    if "values" not in missing:
        group.create_array(
            "values",
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            fill_value=fill_value,
            dimension_names=dims,
            write_data=False,
        )
    if "contigs" not in missing:
        contig_array = group.create_array(
            "contigs",
            shape=(len(contigs),),
            chunks=(max(1, len(contigs)),),
            dtype="str",
            dimension_names=("contig",),
        )
        contig_array[:] = np.asarray(contigs)
    if "offsets" not in missing:
        group.create_array(
            "offsets",
            data=np.asarray(offsets, dtype=offsets_dtype),
            chunks=(max(1, len(offsets)),),
            dimension_names=("contig_boundary",),
        )
    if labels is not None and "labels" not in missing:
        column_dim = dims[1]
        label_array = group.create_array(
            column_dim,
            shape=(len(labels),),
            chunks=(max(1, len(labels)),),
            dtype="str",
            dimension_names=(column_dim,) if label_dims is None else label_dims,
        )
        label_array[:] = np.asarray(labels)
    if extra_array is not None:
        group.create_array(
            extra_array,
            shape=(max(1, shape[0]),),
            chunks=(1,),
            dtype="uint8",
            dimension_names=("quality",),
            write_data=False,
        )

    track_attrs = {
        "zarr_conventions": [{"uuid": "test-perbase", "name": "perbase"}],
        "perbase:version": "0.4",
        "perbase:coordinates": "0-based-half-open",
        "perbase:genome_checksum": _checksum(contigs, offsets),
    }
    if kind is not None:
        track_attrs["perbase:kind"] = kind
    for key, value in (attrs or {}).items():
        if value is None:
            track_attrs.pop(key, None)
        else:
            track_attrs[key] = value
    group.attrs.update(track_attrs)


def _make_track(**kwargs):
    store = CountingMemoryStore()
    group = zarr.open_group(store, mode="w", zarr_format=3)
    _populate_track(group, **kwargs)
    store.reads.clear()
    store.listings.clear()
    return store


def _make_collection(tracks=None, *, nested=None):
    store = CountingMemoryStore()
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_conventions": [{"uuid": "test-perbase", "name": "perbase"}],
            "perbase:kind": "collection",
        }
    )
    for name, kwargs in (tracks or {}).items():
        _populate_track(root.create_group(name), **kwargs)
    if nested is not None:
        _populate_track(root.create_group(nested), chunks=(4,))
    store.reads.clear()
    store.listings.clear()
    return store


def _value_reads(store):
    return [key for key in store.reads if "/values/c/" in f"/{key}"]


def _reads_below(store, group):
    return [key for key in store.reads if key.startswith(f"{group}/")]


def test_open_scalar_track_uses_regular_chunks_without_reading_values():
    store = _make_track(chunks=(4,))

    dataset = pbzarr.open(store)

    assert isinstance(dataset, xr.Dataset)
    assert set(dataset.data_vars) == {"values"}
    assert set(dataset.coords) == {"contigs", "offsets"}
    assert dataset["values"].dims == ("position",)
    assert dataset["values"].dtype == np.dtype("int16")
    assert dataset["values"].chunks == ((4, 4, 2),)
    assert dask.is_dask_collection(dataset["values"])
    assert dataset["contigs"].values.tolist() == ["chr1", "empty", "chr2"]
    assert dataset["offsets"].values.tolist() == [0, 4, 4, 10]
    assert "position" not in dataset.coords
    assert not dataset.xindexes
    assert "source" not in dataset.attrs
    assert "source" not in dataset.encoding
    assert not _value_reads(store)

    source_tasks = [
        task
        for task in dataset["values"].data.__dask_graph__().to_dict().values()
        if hasattr(task, "func")
    ]
    assert source_tasks
    assert all(not task.dependencies for task in source_tasks)

    assert dataset["values"].isel(position=0).compute().item() == -7
    assert _value_reads(store)


def test_open_two_dimensional_track_uses_outer_shards_and_eager_labels():
    store = _make_track(
        shape=(10, 3),
        dims=("position", "sample"),
        labels=("A", "B", "C"),
        chunks=(4, 1),
        shards=(8, 2),
        dtype="uint16",
        fill_value=65535,
    )

    dataset = pbzarr.open(store, chunks={})

    assert dataset["values"].dims == ("position", "sample")
    assert dataset["values"].chunks == ((8, 2), (2, 1))
    assert dataset["values"].dtype == np.dtype("uint16")
    assert dataset["sample"].values.tolist() == ["A", "B", "C"]
    assert not dask.is_dask_collection(dataset["sample"])
    assert set(dataset.coords) == {"contigs", "offsets", "sample"}
    assert not dataset.xindexes
    assert not _value_reads(store)

    assert dataset["values"].isel(position=0, sample=0).compute().item() == 65535


def test_open_with_chunks_none_keeps_the_non_dask_backend_lazy():
    store = _make_track(chunks=(4,))

    dataset = pbzarr.open(store, chunks=None)

    assert dataset["values"].chunks is None
    assert not dask.is_dask_collection(dataset["values"])
    assert not _value_reads(store)
    assert dataset["values"].isel(position=slice(0, 2)).values.tolist() == [-7, -7]
    assert _value_reads(store)


def test_open_with_custom_chunks_uses_normal_xarray_rechunking():
    store = _make_track(
        shape=(10, 3),
        dims=("position", "sample"),
        labels=("A", "B", "C"),
        chunks=(4, 3),
    )

    dataset = pbzarr.open(store, chunks={"position": 6, "sample": 1})

    assert dataset["values"].chunks == ((6, 4), (1, 1, 1))
    assert not _value_reads(store)


def test_open_accepts_an_empty_position_axis():
    store = _make_track(
        contigs=("empty-a", "empty-b"),
        offsets=(0, 0, 0),
        shape=(0,),
        chunks=(1,),
    )

    dataset = pbzarr.open(store)

    assert dataset.sizes["position"] == 0
    assert dataset["values"].chunks == ((0,),)
    assert dataset["offsets"].values.tolist() == [0, 0, 0]
    assert not _value_reads(store)


@pytest.mark.parametrize(
    "fixture",
    [
        {"kind": "mystery"},
        {"attrs": {"perbase:version": "0.3"}},
        {"attrs": {"perbase:coordinates": "1-based-closed"}},
        {"missing": ("values",)},
        {"shape": (2, 10), "dims": ("sample", "position"), "chunks": (1, 4)},
        {"shape": (2, 3, 4), "dims": ("position", "sample", "strand")},
        {"extra_array": "quality"},
        {"contigs": ("", "chr2"), "offsets": (0, 4, 10)},
        {"contigs": ("chr1", "chr1"), "offsets": (0, 4, 10)},
        {"contigs": ("chr1", "chr2"), "offsets": (1, 4, 10)},
        {"contigs": ("chr1", "chr2", "chr3"), "offsets": (0, 7, 5, 10)},
        {"contigs": ("chr1",), "offsets": (0, 4, 10)},
        {"contigs": ("chr1", "chr2"), "offsets": (0, 4, 9), "shape": (10,)},
        {"offsets_dtype": np.float64},
        {"attrs": {"perbase:genome_checksum": "md5:1234"}},
        {
            "attrs": {
                "perbase:genome_checksum": "md5:0123456789abcdef0123456789abcdef"
            }
        },
        {
            "shape": (10, 2),
            "dims": ("position", "sample"),
            "labels": None,
            "chunks": (4, 1),
        },
        {
            "shape": (10, 2),
            "dims": ("position", "sample"),
            "labels": ("A", "A"),
            "chunks": (4, 1),
        },
        {
            "shape": (10, 2),
            "dims": ("position", "region"),
            "labels": ("A", "B"),
            "chunks": (4, 1),
        },
    ],
    ids=[
        "explicit-kind",
        "version",
        "coordinates",
        "missing-values",
        "transposed-values",
        "wrong-rank",
        "unexpected-array",
        "empty-contig",
        "duplicate-contig",
        "offset-start",
        "decreasing-offsets",
        "offset-count",
        "terminal-offset",
        "offset-dtype",
        "malformed-checksum",
        "mismatched-checksum",
        "missing-labels",
        "duplicate-labels",
        "reserved-column-dimension",
    ],
)
def test_invalid_pbz_track_fails_before_reading_values(fixture):
    store = _make_track(**fixture)

    with pytest.raises(pbzarr.PbzError):
        pbzarr.open(store)

    assert not _value_reads(store)


def test_open_infers_a_legacy_track_without_kind():
    store = _make_track(kind=None)

    dataset = pbzarr.open(store)

    assert isinstance(dataset, xr.Dataset)
    assert not _value_reads(store)


def test_open_collection_selects_without_listing_collection_or_touching_others():
    store = _make_collection(
        {
            "depth": {"chunks": (4,)},
            "omitted": {
                "shape": (10, 2),
                "dims": ("position", "sample"),
                "labels": ("A", "B"),
                "chunks": (4, 1),
            },
        }
    )

    tree = pbzarr.open(store, tracks=["depth"])

    assert isinstance(tree, xr.DataTree)
    assert set(tree.children) == {"depth"}
    assert tree["depth"].to_dataset()["values"].chunks == ((4, 4, 2),)
    assert not _reads_below(store, "omitted")
    assert store.listings == ["depth"]
    assert not _value_reads(store)


def test_open_collection_with_empty_selection_returns_only_the_root():
    store = _make_collection({"depth": {"chunks": (4,)}})

    tree = pbzarr.open(store, tracks=[])

    assert not tree.children
    assert tree.attrs["perbase:kind"] == "collection"
    assert not _reads_below(store, "depth")
    assert not store.listings


@pytest.mark.parametrize(
    "tracks",
    [
        [""],
        ["depth", "depth"],
        ["missing"],
        ["collection-child"],
    ],
    ids=[
        "empty",
        "duplicate",
        "unknown",
        "collection-child",
    ],
)
def test_open_collection_rejects_invalid_explicit_children(tracks):
    store = _make_collection(
        {
            "depth": {"chunks": (4,)},
            "collection-child": {"chunks": (4,), "kind": "collection"},
        }
    )

    with pytest.raises(pbzarr.PbzError):
        pbzarr.open(store, tracks=tracks)

    assert not _value_reads(store)


def test_open_all_children_rejects_nested_descendants_before_values():
    store = _make_collection(
        {"depth": {"chunks": (4,)}}, nested="nested/other"
    )

    with pytest.raises(pbzarr.PbzError):
        pbzarr.open(store)

    assert not _value_reads(store)


def test_open_all_children_skips_unpublished_direct_tracks():
    store = _make_collection({"depth": {"chunks": (4,)}})
    root = zarr.open_group(store, mode="a", zarr_format=3)
    pending = root.create_group("pending")
    _populate_track(pending, chunks=(4,))
    pending.attrs.clear()
    store.reads.clear()
    store.listings.clear()

    tree = pbzarr.open(store)

    assert set(tree.children) == {"depth"}
    assert not _value_reads(store)


def test_open_all_children_prepares_each_track_without_indexes():
    store = _make_collection(
        {
            "mask": {"chunks": (4,)},
            "depth": {
                "shape": (10, 2),
                "dims": ("position", "sample"),
                "labels": ("A", "B"),
                "chunks": (4, 1),
            },
        }
    )

    tree = pbzarr.open(store)

    assert set(tree.children) == {"mask", "depth"}
    assert tree["mask"].to_dataset()["values"].chunks == ((4, 4, 2),)
    depth = tree["depth"].to_dataset()
    assert depth["values"].chunks == ((4, 4, 2), (1, 1))
    assert set(depth.coords) == {"contigs", "offsets", "sample"}
    assert all(
        not dask.is_dask_collection(depth[name])
        for name in ("contigs", "offsets", "sample")
    )
    for node in tree.children.values():
        dataset = node.to_dataset()
        assert not dataset.xindexes
        assert not {
            "position",
            "sample",
            "contig",
            "contig_boundary",
        } & set(dataset.xindexes)
    assert not _value_reads(store)


def test_explicit_collection_failure_closes_opened_children(monkeypatch):
    from pbzarr import _open

    store = _make_collection(
        {
            "good": {"chunks": (4,)},
            "bad": {"chunks": (4,), "attrs": {"perbase:version": "0.3"}},
        }
    )
    original_open_zarr = _open._open_zarr
    closed = []

    def tracked_open_zarr(*args, group=None, **kwargs):
        dataset = original_open_zarr(*args, group=group, **kwargs)
        if group is not None:
            def close(name=group):
                closed.append(name)

            dataset.set_close(close)
        return dataset

    monkeypatch.setattr(_open, "_open_zarr", tracked_open_zarr)

    with pytest.raises(pbzarr.PbzError):
        pbzarr.open(store, tracks=["good", "bad"])

    assert closed == ["bad", "good"]


def test_explicit_collection_close_closes_every_child(monkeypatch):
    from pbzarr import _open

    store = _make_collection({"a": {"chunks": (4,)}, "b": {"chunks": (4,)}})
    original_open_zarr = _open._open_zarr
    closed = []

    def tracked_open_zarr(*args, group=None, **kwargs):
        dataset = original_open_zarr(*args, group=group, **kwargs)
        if group is not None:
            def close(name=group):
                closed.append(name)

            dataset.set_close(close)
        return dataset

    monkeypatch.setattr(_open, "_open_zarr", tracked_open_zarr)

    tree = pbzarr.open(store, tracks=["a", "b"])
    assert not closed

    tree.close()

    assert closed == ["b", "a"]


def test_collection_dataset_composes_existing_track_variables_exactly():
    store = _make_collection(
        {
            "mask": {
                "chunks": (4,),
                "attrs": {"perbase:genome_name": "test"},
            },
            "depth": {
                "shape": (10, 2),
                "dims": ("position", "sample"),
                "labels": ("A", "B"),
                "chunks": (4, 1),
                "attrs": {"perbase:genome_name": "test"},
            },
        }
    )
    tree = pbzarr.open(store)

    dataset = tree.pbz.dataset()

    assert set(dataset.data_vars) == {"mask", "depth"}
    assert dataset["mask"].dims == ("position",)
    assert dataset["depth"].dims == ("position", "sample")
    assert set(dataset.coords) == {"contigs", "offsets", "sample"}
    assert dataset["sample"].values.tolist() == ["A", "B"]
    assert dataset.attrs["genome_checksum"] == _checksum(
        ("chr1", "empty", "chr2"), (0, 4, 4, 10)
    )
    assert dataset.attrs["genome_name"] == "test"
    assert dataset.attrs["coordinates"] == "0-based-half-open"
    assert not dataset.xindexes
    assert not _value_reads(store)


@pytest.mark.parametrize(
    "second",
    [
        {
            "contigs": ("chr1", "chr2"),
            "offsets": (0, 5, 10),
            "shape": (10,),
            "chunks": (4,),
        },
        {
            "contigs": ("chr2", "empty", "chr1"),
            "offsets": (0, 6, 6, 10),
            "chunks": (4,),
        },
    ],
    ids=["checksum", "ordered-geometry"],
)
def test_collection_dataset_rejects_incompatible_geometry(second):
    store = _make_collection(
        {"first": {"chunks": (4,)}, "second": second}
    )
    tree = pbzarr.open(store)

    with pytest.raises(pbzarr.PbzError):
        tree.pbz.dataset()

    assert not _value_reads(store)


def test_collection_dataset_rejects_repeated_dimension_label_mismatch():
    common = {
        "shape": (10, 2),
        "dims": ("position", "sample"),
        "chunks": (4, 1),
    }
    store = _make_collection(
        {
            "first": {**common, "labels": ("A", "B")},
            "second": {**common, "labels": ("A", "C")},
        }
    )
    tree = pbzarr.open(store)

    with pytest.raises(pbzarr.PbzError):
        tree.pbz.dataset()

    assert not _value_reads(store)


def test_collection_dataset_rejects_track_name_namespace_collision():
    store = _make_collection(
        {
            "sample": {"chunks": (4,)},
            "depth": {
                "shape": (10, 2),
                "dims": ("position", "sample"),
                "labels": ("A", "B"),
                "chunks": (4, 1),
            },
        }
    )
    tree = pbzarr.open(store)

    with pytest.raises(pbzarr.PbzError):
        tree.pbz.dataset()


def test_collection_dataset_keeps_genome_name_only_when_all_tracks_agree():
    common = {"chunks": (4,), "attrs": {"perbase:genome_name": "test"}}
    same = pbzarr.open(_make_collection({"a": common, "b": common}))
    different = pbzarr.open(
        _make_collection(
            {
                "a": common,
                "b": {
                    "chunks": (4,),
                    "attrs": {"perbase:genome_name": "other"},
                },
            }
        )
    )
    missing = pbzarr.open(
        _make_collection({"a": common, "b": {"chunks": (4,)}})
    )

    assert same.pbz.dataset().attrs["genome_name"] == "test"
    assert "genome_name" not in different.pbz.dataset().attrs
    assert "genome_name" not in missing.pbz.dataset().attrs


def test_collection_dataset_validates_selection_and_allows_empty_selection():
    tree = pbzarr.open(_make_collection({"depth": {"chunks": (4,)}}))

    with pytest.raises(pbzarr.PbzError):
        tree.pbz.dataset(tracks=["missing"])
    with pytest.raises(pbzarr.PbzError):
        tree.pbz.dataset(tracks=["depth", "depth"])

    empty = tree.pbz.dataset(tracks=[])
    assert not empty.variables
    assert empty.attrs["perbase:kind"] == "collection"
