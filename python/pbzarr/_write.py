"""Write an xarray back to a pbz store as a new track.

Single entry point for the common analysis loop: open a store, do some
xarray math, save the result. Infers dtype, column dim, and column
labels from the DataArray(s), creates the track, writes per-contig data,
and refreshes consolidated metadata.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr
import zarr

from ._track import create_track


def write_track(
    path: str,
    track: str,
    data: xr.DataArray | Mapping[str, xr.DataArray],
    *,
    contig: str | None = None,
    overwrite: bool = False,
    **track_kwargs: Any,
) -> None:
    """Write `data` to `path` as a new track.

    `data` may be:
    - an `xr.DataArray` for a single contig (must pass `contig=`),
    - a mapping `{contig_name: xr.DataArray}` for multi-contig writes.

    Each DataArray must have a `position` dim and may have at most one
    other dim, which is treated as the column axis; its coord values
    become the track's column labels and its dim name becomes
    `column_dim` in track metadata.

    `**track_kwargs` are forwarded to `create_track` (e.g. `chunk_size`,
    `shard_size`, `compressors`, `description`, `source`, `fill_value`).
    """
    contigs = _normalize(data, contig)
    if not contigs:
        raise ValueError("write_track: data is empty")

    first_name, first_da = next(iter(contigs.items()))
    dtype, column_dim, columns = _infer_schema(first_name, first_da)

    for cname, da in contigs.items():
        c_dtype, c_dim, c_cols = _infer_schema(cname, da)
        if c_dtype != dtype:
            raise ValueError(
                f"contig {cname!r}: dtype {c_dtype!r} differs from "
                f"{first_name!r} dtype {dtype!r}"
            )
        if c_dim != column_dim or c_cols != columns:
            raise ValueError(
                f"contig {cname!r}: column dim/labels differ from {first_name!r}"
            )

    create_track(
        path,
        track=track,
        dtype=dtype,
        columns=columns,
        column_dim=column_dim,
        overwrite=overwrite,
        **track_kwargs,
    )

    g = zarr.open_group(path, mode="r+")
    np_dtype = np.dtype(dtype)
    for cname, da in contigs.items():
        if cname not in g:
            raise ValueError(f"contig {cname!r} not in store at {path!r}")
        order = ("position",) if column_dim is None else ("position", column_dim)
        arr = np.ascontiguousarray(da.transpose(*order).values, dtype=np_dtype)
        target = g[f"{cname}/{track}"]
        assert isinstance(target, zarr.Array)
        target[:] = arr

    zarr.consolidate_metadata(g.store)


def _normalize(
    data: xr.DataArray | Mapping[str, xr.DataArray],
    contig: str | None,
) -> dict[str, xr.DataArray]:
    if isinstance(data, xr.DataArray):
        if contig is None:
            raise ValueError(
                "write_track: DataArray input requires contig=... "
                "(or pass a {contig: DataArray} mapping)"
            )
        return {contig: data}
    if isinstance(data, Mapping):
        if contig is not None:
            raise ValueError(
                "write_track: contig=... is redundant when data is a mapping"
            )
        for k, v in data.items():
            if not isinstance(v, xr.DataArray):
                raise TypeError(
                    f"write_track: mapping value for {k!r} must be xr.DataArray, "
                    f"got {type(v).__name__}"
                )
        return dict(data)
    raise TypeError(
        f"write_track: data must be xr.DataArray or "
        f"Mapping[str, xr.DataArray], got {type(data).__name__}"
    )


def _infer_schema(
    name: str, da: xr.DataArray
) -> tuple[str, str | None, list[str] | None]:
    if "position" not in da.dims:
        raise ValueError(f"contig {name!r}: DataArray missing 'position' dim")
    non_pos = [d for d in da.dims if d != "position"]
    if len(non_pos) > 1:
        raise ValueError(
            f"contig {name!r}: DataArray has multiple non-position dims "
            f"{non_pos!r}; tracks support at most one column axis"
        )
    if not non_pos:
        return str(da.dtype), None, None
    column_dim = str(non_pos[0])
    if column_dim not in da.coords:
        raise ValueError(
            f"contig {name!r}: column dim {column_dim!r} has no coord labels; "
            f"assign one via .assign_coords({column_dim}=[...])"
        )
    columns = [str(v) for v in da.coords[column_dim].values]
    return str(da.dtype), column_dim, columns
