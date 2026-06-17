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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from ._gather import RegionBlocks
from ._region import RegionQuery, parse_region


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
        query: str | Sequence[str | tuple[str, int, int]],
        *,
        track: str | None = None,
        column: str | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Slice the store to one region, or gather many.

        - `query`: a region string ("chr1:100-200", 0-based half-open; "chr1"
          for the whole contig), OR a sequence of such strings / `(contig,
          start, end)` tuples.

          A single query returns the contiguous slice. A sequence concatenates
          every region's positions along `position`, tagged with an integer
          `region` coord (plus `region_contig` / `region_start`), so the result
          is ready for `.groupby("region")`. Each contig is read once with a
          single gather, so regions sharing chunks are fetched together;
          overlapping regions are handled (positions repeat per region).
        - `track`: optional track name; if given, returns a DataArray.
        - `column`: optional column-axis label (e.g. a sample id for cohort
          tracks, a strand for stranded tracks). Selects on whichever dim the
          track declares as its `column_dim`.
        """
        single = isinstance(query, str)
        queries = [query] if single else list(query)
        rqs = [
            parse_region(q) if isinstance(q, str) else RegionQuery(*q)
            for q in queries
        ]
        for rq in rqs:
            if rq.contig not in self._dt.children:
                available = sorted(self._dt.children)
                raise KeyError(
                    f"contig {rq.contig!r} not in store; contigs: {available}"
                )

        if single:
            rq = rqs[0]
            ds = self._dt[rq.contig].to_dataset()
            n = int(ds.sizes["position"])
            start = rq.start if rq.start is not None else 0
            end = min(rq.end if rq.end is not None else n, n)
            return self._select(ds.isel(position=slice(start, end)), track, column)

        # One chunk-coalesced gather per contig, each position tagged with its
        # region id so the caller can `.groupby("region").reduce(...)`.
        by_contig: dict[str, list[tuple[int, RegionQuery]]] = {}
        for i, rq in enumerate(rqs):
            by_contig.setdefault(rq.contig, []).append((i, rq))

        parts = []
        for contig, items in by_contig.items():
            ds = self._dt[contig].to_dataset()
            n = int(ds.sizes["position"])
            idx, rid, rstart = [], [], []
            for i, rq in items:
                s = rq.start if rq.start is not None else 0
                e = min(rq.end if rq.end is not None else n, n)
                idx.append(np.arange(s, e))
                rid.append(np.full(e - s, i))
                rstart.append(np.full(e - s, s))
            flat = np.concatenate(idx)
            part = ds.isel(position=flat).assign_coords(
                region=("position", np.concatenate(rid)),
                region_contig=("position", np.full(flat.shape[0], contig)),
                region_start=("position", np.concatenate(rstart)),
            )
            parts.append(part)

        combined = parts[0] if len(parts) == 1 else xr.concat(parts, dim="position")
        return self._select(combined, track, column)

    def region_reduced(
        self,
        query: str | Sequence[str | tuple[str, int, int]],
        *,
        track: str,
        reduce: str,
        column: str | None = None,
    ) -> xr.DataArray | xr.Dataset:
        """Collapse each region to one value with a flox-backed `groupby` reduce.

        The optimized form of `region(...).groupby("region").<reduce>()`. `reduce`
        is any xarray groupby reduction (mean, sum, count, max, min, var, std,
        ...). For raw values use `region_blocks`; to stay in xarray use `region`.
        """
        queries = [query] if isinstance(query, str) else list(query)
        da = self.region(queries, track=track, column=column)
        grouped = da.groupby("region")
        try:
            reducer = getattr(grouped, reduce)
        except AttributeError as e:
            raise ValueError(f"unsupported reduce {reduce!r}") from e
        return reducer()

    def region_blocks(
        self,
        query: str | Sequence[str | tuple[str, int, int]],
        *,
        track: str,
        column: str | None = None,
    ) -> RegionBlocks:
        """Return raw per-region values as numpy, aligned to input order.

        Each block is the region's `(n_positions, n_columns)` array (`(n_positions,)`
        for a scalar track); shared column labels and a `(contig, start, end)` table
        come back once. The path for the stat-test loop. For a summary statistic use
        `region_reduced`; to stay in xarray use `region`.
        """
        queries = [query] if isinstance(query, str) else list(query)
        rqs = [
            parse_region(q) if isinstance(q, str) else RegionQuery(*q) for q in queries
        ]
        blocks: list[np.ndarray] = []
        regions: list[tuple[str, int, int]] = []
        columns: np.ndarray | None = None
        for rq in rqs:
            if rq.contig not in self._dt.children:
                raise KeyError(
                    f"contig {rq.contig!r} not in store; "
                    f"contigs: {sorted(self._dt.children)}"
                )
            ds = self._dt[rq.contig].to_dataset()
            n = int(ds.sizes["position"])
            s = rq.start if rq.start is not None else 0
            e = min(rq.end if rq.end is not None else n, n)
            e = max(s, e)
            da = ds[track].isel(position=slice(s, e))
            col_dim = next((d for d in da.dims if d != "position"), None)
            if column is not None and col_dim is not None:
                da = da.sel({col_dim: column})
                col_dim = None
            if col_dim is not None and columns is None:
                columns = np.asarray(da[col_dim].values)
            blocks.append(np.asarray(da.values))
            regions.append((rq.contig, s, e))
        return RegionBlocks(blocks=blocks, columns=columns, regions=regions)

    @staticmethod
    def _select(
        ds: xr.Dataset, track: str | None, column: str | None
    ) -> xr.Dataset | xr.DataArray:
        if track is None:
            return ds
        da = ds[track]
        if column is not None:
            col_dim = next((d for d in da.dims if d != "position"), None)
            if col_dim is not None:
                da = da.sel({col_dim: column})
        return da

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
