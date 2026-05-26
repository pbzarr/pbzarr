"""Parse region strings like 'chr1', 'chr1:100', 'chr1:100-200'."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionQuery:
    contig: str
    start: int | None = None
    end: int | None = None


def parse_region(query: str) -> RegionQuery:
    if ":" not in query:
        return RegionQuery(contig=query)
    contig, _, rest = query.partition(":")
    if "-" not in rest:
        return RegionQuery(contig=contig, start=int(rest))
    s, _, e = rest.partition("-")
    start = int(s) if s else None
    end = int(e) if e else None
    return RegionQuery(contig=contig, start=start, end=end)
