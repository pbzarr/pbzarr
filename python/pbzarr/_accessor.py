"""PBZ operations on native xarray Datasets."""

from __future__ import annotations

import xarray as xr

from ._regions import _resolve_region, regions_dataset


@xr.register_dataset_accessor("pbz")
class PbzDatasetAccessor:
    """PBZ operations for one opened track Dataset."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def region(self, query, *, column=None) -> xr.DataArray:
        """Return one lazy 0-based, half-open contig-local slice."""
        normalized, flat_slice = _resolve_region(self._ds, query)
        result = self._ds["values"].isel(position=flat_slice)
        if column is not None:
            if result.ndim != 2:
                raise ValueError("column= is only valid for a 2D track")
            column_dim = result.dims[1]
            labels = result.coords[column_dim].data.tolist()
            try:
                column_index = labels.index(column)
            except ValueError as error:
                raise KeyError(
                    f"unknown {column_dim} label {column!r}"
                ) from error
            result = result.isel({column_dim: column_index})
        return result.assign_coords(
            region_contig=normalized.contig,
            region_start=normalized.start,
            region_stop=normalized.stop,
        )

    def regions(self, intervals) -> xr.Dataset:
        """Return stable genomic intervals packed along position.

        Dask-backed values remain lazy. With ``chunks=None``, selected pieces
        are gathered eagerly and the complete packed result must fit in RAM.
        """
        return regions_dataset(self._ds, intervals)
