"""One genomic region: the public type, the string parser, and the normalizer."""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass

import numpy as np

from . import _native


@dataclass(frozen=True)
class RegionQuery:
    contig: str
    start: int | None = None
    end: int | None = None


def parse_region(query: str) -> RegionQuery:
    """Parse ``contig``, ``contig:start``, or ``contig:start-end``.

    Coordinates are 0-based and half-open. Commas in coordinates are
    ignored, so ``chr1:1,000-2,000`` is ``chr1:1000-2000``.
    """
    if not isinstance(query, str):
        raise TypeError("region query must be a string")
    return RegionQuery(*_native.parse_region(text=query))


def as_query(query, contigs: Container[str] | None = None) -> RegionQuery:
    """A string, ``RegionQuery`` or ``(contig, start, end)`` tuple as a ``RegionQuery``.

    A string that is a whole name in ``contigs`` is that contig, before any
    split on ``':'``.
    """
    if isinstance(query, RegionQuery):
        region = query
    elif isinstance(query, str):
        region = RegionQuery(*_native.parse_region(text=query, contigs=contigs))
    elif isinstance(query, tuple) and len(query) == 3:
        region = RegionQuery(*query)
    else:
        raise TypeError(
            "region must be a string, RegionQuery, or (contig, start, end) tuple"
        )
    if not isinstance(region.contig, str) or not region.contig:
        raise ValueError("region contig must be a nonempty string")
    bounds = []
    for name in ("start", "end"):
        value = getattr(region, name)
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ):
            raise TypeError(f"region {name} must be an integer or None")
        bounds.append(None if value is None else int(value))
    return RegionQuery(region.contig, *bounds)
