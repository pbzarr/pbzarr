"""Polymorphic open for pbz artifacts."""
from __future__ import annotations


def open(path: str):
    """Open a pbz artifact, returning a `PbzStore` or a `Track` per its
    `perbase:kind` (mirrors Rust `PbzNode::open`).

    For the raw-xarray `DataTree` view, use `store.tree()` on a store handle, or
    `xr.open_datatree(path, engine="zarr")` directly.
    """
    import zarr

    from ._kind import is_perbase, kind_of
    from ._native import PbzError
    from ._pbzstore import PbzStore
    from ._track import Track

    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    if not is_perbase(attrs):
        raise PbzError(f"{path!r} is not a pbz artifact (no zarr_conventions marker)")
    if kind_of(attrs) == "track":
        return Track.open(path)
    return PbzStore(path)
