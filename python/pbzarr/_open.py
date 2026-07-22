"""Thin wrapper around xr.open_datatree for pbz stores."""
from __future__ import annotations
from typing import Any

import xarray as xr


def open(path: str, *, chunks: Any = None) -> xr.DataTree:
    """Open a pbz store as an `xr.DataTree` (the xarray-flavored entry point).

    Defaults to eager (matches `xr.open_datatree`'s convention). Pass
    `chunks={}` (or any chunk dict) for a dask-backed DataTree where
    dask chunks align with on-disk zarr chunks.

    For the optimized read -> transform -> write pipeline, use `PbzStore`
    which defaults to lazy. `pbzarr.open` exists for users who want plain
    xarray semantics without the store handle.
    """
    if chunks is None:
        return xr.open_datatree(path, engine="zarr", consolidated=False)
    return xr.open_datatree(path, engine="zarr", chunks=chunks, consolidated=False)
