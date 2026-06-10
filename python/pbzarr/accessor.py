"""xarray accessor `.pbz` for pbz stores.

Registered on `xr.DataTree`. Provides region queries (string-based,
0-based half-open) and a list of tracks from root metadata. Column-axis
selection uses a generic `column` kwarg that resolves against whatever
the track declared as its `column_dim` — pbzarr is not cohort-specific,
so the API never hardcodes `"sample"`.

The accessor also provides `assign_column_labels`, which papers over a
real xarray gap: non-dim coords do not auto-propagate from DataTree
root to children (xarray #9472), so the per-child loop is unavoidable
for the common case of "attach a label like `pop` to every contig's
column axis."
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from ._region import parse_region


@xr.register_datatree_accessor("pbz")
class PbzDataTreeAccessor:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt

    @property
    def tracks(self) -> list[str]:
        """Track names from the root perbase_zarr.tracks map."""
        attrs = self._dt.attrs.get("perbase_zarr", {})
        return sorted(attrs.get("tracks", {}).keys())

    def region(
        self,
        query: str,
        *,
        track: str | None = None,
        column: str | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Slice the store to a region.

        - `query`: string like "chr1:100-200" (0-based, half-open).
                   "chr1" returns the whole contig.
        - `track`: optional track name; if given, returns the sliced DataArray.
        - `column`: optional column-axis label (e.g. a sample id for cohort
                    tracks, a strand for stranded tracks). Selects on whichever
                    dim the track declares as its `column_dim`.
        """
        rq = parse_region(query)
        if rq.contig not in self._dt.children:
            available = sorted(self._dt.children)
            raise KeyError(
                f"contig {rq.contig!r} not in store; contigs: {available}"
            )
        ds = self._dt[rq.contig].to_dataset()

        n = int(ds.sizes["position"])
        start = rq.start if rq.start is not None else 0
        end = rq.end if rq.end is not None else n
        end = min(end, n)
        ds = ds.isel(position=slice(start, end))

        if track is not None:
            da = ds[track]
            if column is not None:
                col_dim = next((d for d in da.dims if d != "position"), None)
                if col_dim is not None:
                    da = da.sel({col_dim: column})
            return da
        return ds

    def assign_column_labels(
        self,
        dim: str,
        **labels: Mapping[Any, Any],
    ) -> xr.DataTree:
        """Attach non-dim coords on `dim` to every child Dataset.

        Each kwarg becomes a new non-dim coord on `dim`, with values looked
        up in the provided dict from the existing dim values. Children that
        don't have `dim` are passed through unchanged. Missing keys raise
        before any work happens, instead of `KeyError`ing mid-compute the
        way a lambda would.

        Example:
            labeled = depth.pbz.assign_column_labels(
                "sample",
                pop={"s1": "POP_A", "s2": "POP_B"},
                sex={"s1": "F",     "s2": "M"},
            )
        """
        if not labels:
            return self._dt

        def _apply(ds: xr.Dataset) -> xr.Dataset:
            if dim not in ds.dims:
                return ds
            dim_values = ds[dim].values
            new_coords: dict[str, Any] = {}
            for label_name, mapping in labels.items():
                missing = [v for v in dim_values if v not in mapping]
                if missing:
                    raise ValueError(
                        f"assign_column_labels: label {label_name!r} missing "
                        f"keys for dim {dim!r}: {missing}"
                    )
                new_coords[label_name] = (
                    dim,
                    [mapping[v] for v in dim_values],
                )
            return ds.assign_coords(**new_coords)

        return self._dt.map_over_datasets(_apply)
