"""Reference geometry: contig names and lengths, and the rows they map to."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cached_property
import re

import numpy as np

from ._region_query import RegionQuery, as_query

_I64_MAX = np.iinfo(np.int64).max
_BIN_DIM = re.compile(r"bin_([1-9][0-9]*)\Z")


def bin_width(dim: str) -> int | None:
    """Width N of a ``bin_N`` dimension, 1 for ``position``, else None.

    The ``bin_`` prefix is reserved: a name such as ``bin_01`` is refused.
    """
    if dim == "position":
        return 1
    if not dim.startswith("bin_"):
        return None
    match = _BIN_DIM.match(dim)
    if match is None or int(match.group(1)) > _I64_MAX:
        raise ValueError(f"{dim!r} is not a canonical bin_N dimension name")
    return int(match.group(1))


class Reference(Mapping[str, int]):
    """Ordered contig lengths; the genomic axes are their concatenation.

    Coordinates are 0-based and half-open within each contig. A ``bin_N``
    axis restarts at every contig, so a contig's last bin can be short.
    """

    def __init__(self, names, lengths) -> None:
        names = tuple(names)
        lengths = np.asarray(lengths)
        if lengths.ndim != 1 or lengths.size != len(names):
            raise ValueError("reference names and lengths must have equal length")
        if lengths.size and (
            lengths.dtype.kind not in "iu"
            or np.any(lengths < 0)
            or int(lengths.max()) > _I64_MAX
        ):
            raise ValueError("reference lengths must be nonnegative integers")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("reference names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("reference names must be unique")
        if sum(int(n) for n in lengths.tolist()) > _I64_MAX:
            raise ValueError("reference extent exceeds signed 64-bit indexing")
        lengths = lengths.astype(np.int64)
        lengths.setflags(write=False)
        self._names = names
        self._lengths = lengths
        self._index = {name: i for i, name in enumerate(names)}

    @classmethod
    def from_xarray(cls, obj) -> Reference:
        """The reference of a DataArray or Dataset with a ``reference`` coordinate."""
        from ._coord import reference_of

        return reference_of(obj)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @cached_property
    def offsets(self) -> np.ndarray:
        """Exclusive prefix sum of lengths, one more entry than contigs."""
        return self._offsets(1)

    def _offsets(self, width: int) -> np.ndarray:
        counts = -(-self._lengths // width) if width > 1 else self._lengths
        offsets = np.zeros(counts.size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        offsets.setflags(write=False)
        return offsets

    def _axis_offsets(self, dim: str) -> tuple[int, np.ndarray]:
        width = bin_width(dim)
        if width is None:
            raise ValueError(f"{dim!r} is not a genomic dimension")
        return width, self.offsets if width == 1 else self._offsets(width)

    def size(self, dim: str = "position") -> int:
        """Extent of a genomic dimension: ``position`` or ``bin_N``."""
        return int(self._axis_offsets(dim)[1][-1])

    def resolve(self, query) -> RegionQuery:
        """One region with contig-local ``start`` and ``end`` filled in.

        ``query`` takes the forms of ``rows``. An open start is 0 and an open
        end is the contig end. Empty and out-of-range regions are refused.
        """
        region = as_query(query, self._index)
        if region.contig not in self._index:
            raise KeyError(f"unknown contig {region.contig!r}")
        length = int(self._lengths[self._index[region.contig]])
        start = 0 if region.start is None else region.start
        end = length if region.end is None else region.end
        if start < 0 or start >= end or end > length:
            raise ValueError(
                f"{region.contig}:[{start},{end}) is empty or outside length {length}"
            )
        return RegionQuery(region.contig, start, end)

    def rows(self, query, dim: str = "position") -> slice:
        """Flat rows of ``dim`` covering one region.

        ``query`` is ``"chr1"``, ``"chr1:1000"`` (to the contig end),
        ``"chr1:1000-2000"``, a ``RegionQuery`` or a ``(contig, start, end)``
        tuple. On ``bin_N`` the rows are the bins that intersect the region.
        Empty and out-of-range regions are refused.
        """
        region = self.resolve(query)
        width, offsets = self._axis_offsets(dim)
        base = int(offsets[self._index[region.contig]])
        return slice(base + region.start // width, base + -(-region.end // width))

    def interval(self, row: int, dim: str = "position") -> RegionQuery:
        """The contig-local interval one row of ``dim`` covers."""
        width, offsets = self._axis_offsets(dim)
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, (int, np.integer)):
            raise TypeError("row must be an integer")
        row = int(row)
        if row < 0 or row >= int(offsets[-1]):
            raise ValueError(f"row {row} is outside {dim}")
        index = int(np.searchsorted(offsets, row, side="right")) - 1
        start = (row - int(offsets[index])) * width
        end = min(start + width, int(self._lengths[index]))
        return RegionQuery(self._names[index], start, end)

    def __getitem__(self, name: str) -> int:
        return int(self._lengths[self._index[name]])

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        shown = ", ".join(f"{n}: {self[n]}" for n in self._names[:4])
        more = ", ..." if len(self._names) > 4 else ""
        return f"Reference({{{shown}{more}}})"
