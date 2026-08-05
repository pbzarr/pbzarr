"""Region resolution over the flat layout: contig region -> flat slice.

Pure functions over zarr-python + numpy + xarray. Each track is one flat
`values` array with a per-track `offsets` prefix-sum index over `contigs`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
import zarr

from ._region import RegionQuery


@dataclass
class RegionBlocks:
    blocks: list[np.ndarray]
    columns: np.ndarray | None
    regions: list[tuple[str, int, int]]


@dataclass
class _TrackArrays:
    values: "zarr.Array"
    offsets: np.ndarray
    contigs: list[str]
    dims: tuple[str, ...]
    col_dim: str | None
    labels: list[str] | None


def open_track(store_path: str, name: str) -> _TrackArrays:
    values = zarr.open_array(f"{store_path}/{name}/values", mode="r")
    dims = tuple(values.metadata.dimension_names)
    offsets = np.asarray(zarr.open_array(f"{store_path}/{name}/offsets", mode="r")[:])
    contigs = [str(c) for c in zarr.open_array(f"{store_path}/{name}/contigs", mode="r")[:]]
    col_dim = dims[1] if len(dims) == 2 else None
    labels = None
    if col_dim is not None:
        labels = [str(x) for x in zarr.open_array(f"{store_path}/{name}/{col_dim}", mode="r")[:]]
    return _TrackArrays(values, offsets, contigs, dims, col_dim, labels)


def _resolve(ta: _TrackArrays, rq: RegionQuery) -> tuple[int, int, int]:
    try:
        i = ta.contigs.index(rq.contig)
    except ValueError as e:
        raise KeyError(f"contig {rq.contig!r} not in track; contigs: {ta.contigs}") from e
    base = int(ta.offsets[i])
    clen = int(ta.offsets[i + 1]) - base
    s = rq.start if rq.start is not None else 0
    e = clen if rq.stop is None else min(rq.stop, clen)
    e = max(s, e)
    return base, s, e


def _slice(ta: _TrackArrays, contig: str, base: int, s: int, e: int, column) -> xr.DataArray:
    lo, hi = base + s, base + e
    pos = np.arange(s, e)
    if ta.col_dim is None:
        da = xr.DataArray(np.asarray(ta.values[lo:hi]), dims=["position"],
                          coords={"position": pos})
    else:
        da = xr.DataArray(np.asarray(ta.values[lo:hi, :]), dims=["position", ta.col_dim],
                          coords={"position": pos, ta.col_dim: ta.labels})
        if column is not None:
            da = da.sel({ta.col_dim: column})
    return da.assign_coords(contig=contig)


def read_region(store_path: str, name: str, rq: RegionQuery, column=None) -> xr.DataArray:
    ta = open_track(store_path, name)
    base, s, e = _resolve(ta, rq)
    return _slice(ta, rq.contig, base, s, e, column)


def gather_regions(store_path: str, name: str, rqs: list[RegionQuery], column=None) -> xr.DataArray:
    ta = open_track(store_path, name)
    parts = []
    for idx, rq in enumerate(rqs):
        base, s, e = _resolve(ta, rq)
        da = _slice(ta, rq.contig, base, s, e, column)
        n = e - s
        da = da.assign_coords(
            region=("position", np.full(n, idx)),
            region_contig=("position", np.full(n, rq.contig)),
            region_start=("position", np.full(n, s)),
        )
        parts.append(da)
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim="position", coords="different", compat="equals")


def region_blocks(store_path: str, name: str, rqs: list[RegionQuery], column=None) -> RegionBlocks:
    ta = open_track(store_path, name)
    blocks: list[np.ndarray] = []
    regions: list[tuple[str, int, int]] = []
    columns: np.ndarray | None = None
    for rq in rqs:
        base, s, e = _resolve(ta, rq)
        lo, hi = base + s, base + e
        if ta.col_dim is None:
            arr = np.asarray(ta.values[lo:hi])
        else:
            arr = np.asarray(ta.values[lo:hi, :])
            if column is not None:
                arr = arr[:, ta.labels.index(column)]
            elif columns is None:
                columns = np.asarray(ta.labels)
        blocks.append(arr)
        regions.append((rq.contig, s, e))
    return RegionBlocks(blocks=blocks, columns=columns, regions=regions)
