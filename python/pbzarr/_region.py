"""Canonical public representation and parser for one genomic region."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionQuery:
    contig: str
    start: int | None = None
    end: int | None = None


def parse_region(query: str) -> RegionQuery:
    """Parse ``contig``, ``contig:start``, or ``contig:start-end``."""
    if not isinstance(query, str):
        raise TypeError("region query must be a string")
    query = query.strip()
    if not query:
        raise ValueError("region contig must be a nonempty string")
    if ":" not in query:
        return RegionQuery(contig=query)

    contig, coordinates = query.rsplit(":", 1)
    if not contig:
        raise ValueError("region contig must be a nonempty string")
    if not coordinates:
        return RegionQuery(contig=contig)
    if "-" not in coordinates:
        return RegionQuery(contig=contig, start=_parse_coordinate(coordinates))

    start_text, end_text = coordinates.split("-", 1)
    start = _parse_coordinate(start_text) if start_text else None
    end = _parse_coordinate(end_text) if end_text else None
    return RegionQuery(contig=contig, start=start, end=end)


def _parse_coordinate(value: str) -> int:
    if not value.isdecimal():
        raise ValueError(f"invalid region coordinate {value!r}")
    return int(value)
