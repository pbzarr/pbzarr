"""Thin wrapper around xr.open_datatree for pbz stores."""
from __future__ import annotations
import xarray as xr


def open(path: str, *, chunks: dict | str | None = None) -> xr.DataTree:
    """Open a pbz store as an `xr.DataTree`.

    Each contig is a child node holding the per-contig data arrays + coord
    arrays. Defaults to **eager numpy backing** — for cohort-scale reads
    the zarr codec pipeline (especially via `zarrs-python`) parallelizes
    chunk decode internally and is faster than layering dask on top
    (see https://github.com/zarrs/zarr_benchmarks).

    Pass `chunks="auto"` or an explicit dict for dask-backed lazy reads
    if you specifically need out-of-core streaming.

    Use `dt.pbz.region(...)` for region queries; the accessor is registered
    when `pbzarr` is imported.
    """
    return xr.open_datatree(path, engine="zarr", chunks=chunks)
