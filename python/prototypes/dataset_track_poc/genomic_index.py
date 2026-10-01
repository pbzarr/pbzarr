"""Throwaway compact genomic index; not part of the production PBZ API.

Coordinate storage uses Xarray's public IndexVariable with pandas.RangeIndex.
Only slice-based positional selection is supported; no dense genome coordinate
is created here. Native consumers can still explicitly materialize coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate
from numbers import Integral
import re

import pandas as pd
import xarray as xr
from xarray.core.indexing import IndexSelResult  # Private Xarray API: PoC only.


def genomic_width(dim: str) -> int | None:
    if dim == "position":
        return 1
    if isinstance(dim, str) and dim.startswith("bin_"):
        if not re.fullmatch(r"bin_[1-9][0-9]*", dim):
            raise ValueError(f"Invalid genomic dimension: {dim!r}")
        return int(dim[4:])
    return None


@dataclass(frozen=True)
class Reference:
    names: tuple[str, ...]
    lengths: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "lengths", tuple(self.lengths))
        if len(self.names) != len(self.lengths):
            raise ValueError("Reference names and lengths must have equal length")
        if any(not isinstance(n, str) or not n for n in self.names):
            raise ValueError("Reference names must be nonempty strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Reference names must be unique")
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
               for n in self.lengths):
            raise ValueError("Reference lengths must be nonnegative integers")
        object.__setattr__(self, "lengths", tuple(map(int, self.lengths)))
        if sum(self.lengths) > 2**63 - 1:
            raise ValueError("Reference extent exceeds signed 64-bit indexing")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> Reference:
        return cls(tuple(mapping), tuple(mapping.values()))

    def offsets(self, width: int = 1) -> tuple[int, ...]:
        if isinstance(width, bool) or not isinstance(width, Integral) or width <= 0:
            raise ValueError("Bin width must be a positive integer")
        width = int(width)
        return tuple(accumulate(((n + width - 1) // width for n in self.lengths), initial=0))

    def size(self, width: int = 1) -> int:
        return self.offsets(width)[-1]

    def region_slice(self, query, width: int = 1) -> slice:
        """Return full-reference row bounds; bins overlap the half-open query."""
        if isinstance(query, str):
            match = re.fullmatch(r"(.+):([0-9]+)-([0-9]+)", query)
            if match is None:
                raise ValueError("Use a zero-based 'contig:start-end' region")
            contig, start, stop = match.groups()
            start, stop = int(start), int(stop)
        elif isinstance(query, tuple) and len(query) == 3:
            contig, start, stop = query
        else:
            raise TypeError("A region must be 'contig:start-end' or a three-item tuple")
        try:
            contig_index = self.names.index(contig)
        except ValueError as error:
            raise KeyError(f"Unknown contig: {contig!r}") from error
        if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (start, stop)):
            raise ValueError("Region bounds must be integers")
        if not 0 <= start <= stop <= self.lengths[contig_index]:
            raise ValueError(f"Region bounds are outside {contig!r}")
        offset = self.offsets(width)[contig_index]
        start, stop, width = int(start), int(stop), int(width)
        first = offset + start // width
        last = first if start == stop else offset + (stop + width - 1) // width
        return slice(first, last)


class GenomicIndex(xr.Index):
    """Reference plus compact canonical row IDs, carried by native Xarray."""

    def __init__(self, reference: Reference, dim: str = "position", rows: range | None = None):
        width = genomic_width(dim)
        if width is None:
            raise ValueError(f"Not a genomic dimension: {dim!r}")
        self.reference, self.dim, self.width = reference, dim, width
        self.rows = range(reference.size(width)) if rows is None else rows
        if not isinstance(self.rows, range):
            raise TypeError("This PoC represents genomic rows with a range only")
        if self.rows and not (0 <= min(self.rows[0], self.rows[-1])
                              <= max(self.rows[0], self.rows[-1]) < reference.size(width)):
            raise ValueError("Selected genomic rows are outside the reference")

    @classmethod
    def from_variables(cls, variables, *, options):
        if len(variables) != 1:
            raise ValueError("A genomic index owns exactly one coordinate")
        dim, variable = next(iter(variables.items()))
        if variable.dims != (dim,) or not isinstance(variable, xr.IndexVariable):
            raise ValueError("Expected a one-dimensional compact index coordinate")
        labels = variable.to_index()
        if not isinstance(labels, pd.RangeIndex):
            raise ValueError("This PoC only accepts pandas.RangeIndex coordinates")
        return cls(options["reference"], dim, range(labels.start, labels.stop, labels.step))

    def to_pandas_index(self):
        return pd.RangeIndex(self.rows.start, self.rows.stop, self.rows.step, name=self.dim)

    def create_variables(self, variables=None):
        old = variables.get(self.dim) if variables is not None else None
        return {self.dim: xr.IndexVariable(
            self.dim, self.to_pandas_index(),
            attrs=old.attrs if old is not None else None,
            encoding=old.encoding if old is not None else None,
        )}

    def isel(self, indexers):
        indexer = indexers.get(self.dim, slice(None))
        if isinstance(indexer, slice):
            return type(self)(self.reference, self.dim, self.rows[indexer])
        if isinstance(indexer, Integral) and not isinstance(indexer, bool):
            return None  # The genomic axis was removed.
        if isinstance(indexer, xr.Variable) and indexer.ndim == 0:
            return None
        raise NotImplementedError("This PoC supports genomic slices, not fancy indexing")

    def sel(self, labels, method=None, tolerance=None):
        if method is not None or tolerance is not None:
            raise ValueError("Genomic selection does not support nearest/tolerance")
        bounds = self.reference.region_slice(labels[self.dim], self.width)
        rows, n = self.rows, len(self.rows)
        if rows.step > 0:
            first = -((rows.start - bounds.start) // rows.step)
            last = -((rows.start - bounds.stop) // rows.step)
        else:
            step = -rows.step
            first = (rows.start - bounds.stop) // step + 1
            last = (rows.start - bounds.start) // step + 1
        first = min(n, max(0, first))
        last = min(n, max(first, last))
        return IndexSelResult({self.dim: slice(first, last)})

    def equals(self, other, **kwargs):
        return (isinstance(other, GenomicIndex) and self.reference == other.reference
                and self.dim == other.dim and self.rows == other.rows)

    def join(self, other, how="inner"):
        if not self.equals(other):
            raise ValueError("Genomic indexes must match exactly; implicit alignment is disabled")
        return self

    def reindex_like(self, other, **kwargs):
        self.join(other)
        return {}  # Equal indexes need no dense arange indexer.

    def rename(self, name_dict, dims_dict):
        if name_dict.get(self.dim, self.dim) != self.dim or dims_dict.get(self.dim, self.dim) != self.dim:
            raise ValueError("Renaming genomic dimensions is unsupported in this PoC")
        return self


def attach_indexes(ds: xr.Dataset, reference: Reference) -> xr.Dataset:
    """Attach indexes only to complete canonical genomic dimensions."""
    for dim, extent in ds.sizes.items():
        width = genomic_width(dim)
        if width is None:
            continue
        if extent != reference.size(width):
            raise ValueError(f"{dim!r} does not span the full reference")
        index = GenomicIndex(reference, dim)
        existing = ds.xindexes.get(dim)
        if existing is not None and index.equals(existing):
            continue
        if dim in ds.coords:
            raise ValueError(f"Refusing to replace an unverified {dim!r} coordinate")
        ds = ds.assign_coords(xr.Coordinates.from_xindex(index))
    return ds


def get_index(array: xr.DataArray) -> GenomicIndex | None:
    dims = [dim for dim in array.dims if genomic_width(dim) is not None]
    if len(dims) > 1:
        raise ValueError("A track cannot have multiple genomic dimensions")
    if not dims:
        return None
    index = array.xindexes.get(dims[0])
    if not isinstance(index, GenomicIndex):
        raise ValueError("Genomic context is missing; retain or explicitly rebuild the PBZ index")
    return index
