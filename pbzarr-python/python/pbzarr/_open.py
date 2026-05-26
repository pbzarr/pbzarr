"""Thin wrapper around xr.open_datatree for pbz stores."""
from __future__ import annotations
import xarray as xr


def open(path: str) -> xr.DataTree:
    """Open a pbz store as an `xr.DataTree`.

    The root node holds `contigs` / `contig_lengths` arrays and the
    `perbase_zarr` attribute namespace; each contig is a child node
    holding the per-contig data arrays + coord arrays.

    Use `dt.pbz.region(...)` for region queries; the accessor is
    registered when `pbzarr` is imported.
    """
    return xr.open_datatree(path, engine="zarr")
