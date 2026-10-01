"""Throwaway contig-aware coarsening, backed by ordinary Xarray reductions.

The experiment registers ``.pbz_poc.coarsen(...)`` on native DataArrays. A
production API could expose the same operation through ``.pbz``; this module
deliberately leaves that accessor and production code alone.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import xarray as xr

from .genomic_index import GenomicIndex, get_index


class GenomicCoarsen:
    """Reduce full-reference base data into bins anchored at each contig's zero.

    This proof of concept requires a canonical, complete ``position`` index and
    positive contig lengths. Regional views, empty references, and zero-length
    contigs are explicitly outside its scope. A final short bin contains only
    the contig's remaining bases, with no artificial padding.
    """

    def __init__(self, array: xr.DataArray, *, position: int):
        if isinstance(position, bool):
            raise ValueError("The bin width must be a positive integer.")
        try:
            width = operator.index(position)
        except TypeError as exc:
            raise ValueError("The bin width must be a positive integer.") from exc
        if width <= 0:
            raise ValueError("The bin width must be a positive integer.")

        index = get_index(array)
        if index is None or index.dim != "position":
            raise ValueError("This coarsening prototype requires indexed base data.")
        reference = index.reference
        if not reference.lengths or any(length <= 0 for length in reference.lengths):
            raise ValueError(
                "This prototype requires nonempty, positive-length contigs."
            )
        if index.rows != range(reference.size(1)):
            raise ValueError(
                "This prototype requires the full reference in canonical order."
            )

        self._array = array
        self._reference = reference
        self._width = width

    def mean(self, **kwargs: Any) -> xr.DataArray:
        return self._reduce("mean", **kwargs)

    def sum(self, **kwargs: Any) -> xr.DataArray:
        return self._reduce("sum", **kwargs)

    def min(self, **kwargs: Any) -> xr.DataArray:
        return self._reduce("min", **kwargs)

    def max(self, **kwargs: Any) -> xr.DataArray:
        return self._reduce("max", **kwargs)

    def reduce(self, func: Callable, **kwargs: Any) -> xr.DataArray:
        """Delegate a native ``func(array, axis=..., **kwargs)`` reduction."""
        return self._reduce("reduce", func=func, **kwargs)

    def _reduce(
        self, operation: str, *, func: Callable | None = None, **kwargs: Any
    ) -> xr.DataArray:
        kwargs.setdefault("keep_attrs", False)
        if {"dim", "axis", "keepdims"}.intersection(kwargs):
            raise TypeError(
                "The genomic coarsener fixes the reduction axis and output shape."
            )

        dim = f"bin_{self._width}"
        # The index describes positions; it is not a numerical coordinate to
        # average. Other position-dependent annotations have no generic bin
        # reduction, so retain only non-genomic coordinates in this experiment.
        position_coords = [
            name
            for name, coord in self._array.coords.items()
            if "position" in coord.dims
        ]
        data = self._array.drop_vars(position_coords)
        pieces = []
        offsets = self._reference.offsets(1)

        for start, end in zip(offsets[:-1], offsets[1:]):
            full_end = start + ((end - start) // self._width) * self._width
            if full_end > start:
                windows = data.isel(position=slice(start, full_end)).coarsen(
                    position=self._width, boundary="exact"
                )
                if operation == "reduce":
                    reduced = windows.reduce(func, **kwargs)
                else:
                    reduced = getattr(windows, operation)(**kwargs)
                pieces.append(reduced.rename(position=dim))

            if full_end < end:
                tail = data.isel(position=slice(full_end, end))
                if operation == "reduce":
                    reduced = tail.reduce(func, dim="position", **kwargs)
                else:
                    reduced = getattr(tail, operation)(dim="position", **kwargs)
                pieces.append(reduced.expand_dims({dim: 1}))

        result = xr.concat(
            pieces, dim=dim, coords="minimal", compat="equals", join="exact"
        )
        result = result.transpose(dim, *(d for d in data.dims if d != "position"))
        index = GenomicIndex(
            self._reference,
            dim=dim,
            rows=range(self._reference.size(self._width)),
        )
        return result.assign_coords(xr.Coordinates.from_xindex(index))


@xr.register_dataarray_accessor("pbz_poc")
class _GenomicAccessor:
    """Native Xarray entry point to the same coarsener used by Track."""

    def __init__(self, array: xr.DataArray):
        self._array = array

    def coarsen(self, *, position: int) -> GenomicCoarsen:
        return GenomicCoarsen(self._array, position=position)
