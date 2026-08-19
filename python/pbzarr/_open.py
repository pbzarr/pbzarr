"""Open PBZ nodes through xarray's native Zarr backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack

import xarray as xr
import zarr
from zarr.errors import GroupNotFoundError

from ._native import PbzError
from ._xarray import (
    _NATIVE_CHUNKS,
    _has_perbase_marker,
    _node_kind,
    _normalize_track_names,
    _open_zarr,
    _prepare_track_dataset,
)


def _open_selected_collection(
    source,
    root_attrs: Mapping,
    names: list[str],
    *,
    chunks,
    storage_options,
) -> xr.DataTree:
    with ExitStack() as stack:
        datasets = {"/": xr.Dataset(attrs=root_attrs)}
        for name in names:
            try:
                opened = _open_zarr(
                    source,
                    group=name,
                    storage_options=storage_options,
                )
            except GroupNotFoundError as error:
                raise PbzError(f"unknown track {name!r}") from error
            dataset = _prepare_track_dataset(
                opened,
                chunks=chunks,
                source=source,
                group=name,
                storage_options=storage_options,
            )
            stack.callback(dataset.close)
            datasets[f"/{name}"] = dataset

        tree = xr.DataTree.from_dict(datasets)
        tree.set_close(stack.pop_all().close)
        return tree


def _read_root_attrs(source, *, storage_options) -> dict:
    kwargs = {}
    if storage_options is not None:
        kwargs["storage_options"] = dict(storage_options)
    group = zarr.open_group(
        source,
        mode="r",
        use_consolidated=False,
        **kwargs,
    )
    try:
        return dict(group.attrs)
    finally:
        caller_store = getattr(source, "store", source)
        if group.store is not caller_store:
            group.store.close()


def _open_all_collection(
    source, *, chunks, storage_options
) -> xr.DataTree:
    kwargs = {}
    if storage_options is not None:
        kwargs["storage_options"] = dict(storage_options)
    opened = xr.open_datatree(
        source,
        engine="zarr",
        chunks=None,
        consolidated=False,
        create_default_indexes=False,
        decode_cf=False,
        **kwargs,
    )
    try:
        if _node_kind(opened.attrs) != "collection":
            raise PbzError("PBZ node is a track, not a collection")
        published = {
            name: node
            for name, node in opened.children.items()
            if _has_perbase_marker(node.attrs)
        }
        # Published tracks may carry subtrees of their own (a scales/
        # pyramid); deep descendants are rejected only outside them.
        if any(
            node.path.count("/") > 1 and node.path.split("/", 2)[1] not in published
            for node in opened.descendants
        ):
            raise PbzError("PBZ collections may contain only direct track children")
        for node in published.values():
            if _node_kind(node.attrs) != "track":
                raise PbzError("PBZ collection children must be tracks")

        datasets = {"/": opened.to_dataset(inherit=False)}
        for name, node in published.items():
            datasets[f"/{name}"] = _prepare_track_dataset(
                node.to_dataset(inherit=False),
                chunks=chunks,
                source=source,
                group=name,
                storage_options=storage_options,
            )
        tree = xr.DataTree.from_dict(datasets)
        tree.set_close(opened.close)
        return tree
    except Exception:
        opened.close()
        raise


def open(
    source,
    *,
    tracks: Sequence[str] | None = None,
    chunks=_NATIVE_CHUNKS,
    storage_options: Mapping[str, object] | None = None,
) -> xr.Dataset | xr.DataTree:
    """Open a PBZ track as a Dataset or a collection as a DataTree."""
    root_attrs = _read_root_attrs(source, storage_options=storage_options)
    kind = _node_kind(root_attrs)
    if kind == "track":
        if tracks is not None:
            raise PbzError("tracks= is only valid for a PBZ collection")
        root = _open_zarr(source, storage_options=storage_options)
        return _prepare_track_dataset(
            root,
            chunks=chunks,
            source=source,
            storage_options=storage_options,
        )
    if tracks is not None:
        names = _normalize_track_names(tracks)
        return _open_selected_collection(
            source,
            root_attrs,
            names,
            chunks=chunks,
            storage_options=storage_options,
        )
    return _open_all_collection(
        source,
        chunks=chunks,
        storage_options=storage_options,
    )
