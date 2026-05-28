"""xarray accessor `.pbz` for pbz stores.

Registered on `xr.DataTree`. Provides region queries (string-based,
0-based half-open) and a list of tracks from root metadata. Column-axis
selection uses a generic `column` kwarg that resolves against whatever
the track declared as its `column_dim` — pbzarr is not cohort-specific,
so the API never hardcodes `"sample"`.
"""
from __future__ import annotations
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
