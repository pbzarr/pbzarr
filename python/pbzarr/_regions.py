"""Normalization and geometry resolution for PBZ region operations."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._native import PbzError
from ._region import RegionQuery, parse_region
from ._xarray import _validate_track


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _normalize_one(query) -> RegionQuery:
    if isinstance(query, RegionQuery):
        normalized = query
    elif isinstance(query, str):
        normalized = parse_region(query)
    elif isinstance(query, tuple) and len(query) == 3:
        normalized = RegionQuery(query[0], query[1], query[2])
    else:
        raise TypeError(
            "region query must be a RegionQuery, string, or "
            "(contig, start, stop) tuple"
        )

    if not isinstance(normalized.contig, str) or not normalized.contig:
        raise ValueError("region contig must be a nonempty string")
    return RegionQuery(
        normalized.contig,
        _normalize_coordinate(normalized.start, "start"),
        _normalize_coordinate(normalized.stop, "stop"),
    )


def _normalize_coordinate(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"region {name} must be an integer or None")
    coordinate = int(value)
    if coordinate < _I64_MIN or coordinate > _I64_MAX:
        raise ValueError(f"region {name} is outside signed 64-bit range")
    return coordinate


def _resolve_region(
    ds: xr.Dataset, query
) -> tuple[RegionQuery, slice]:
    if "pbz:representation" in ds.attrs:
        raise PbzError("region() requires a normal PBZ track Dataset")
    _validate_track(ds)
    normalized = _normalize_one(query)
    contigs = np.asarray(ds["contigs"].values).tolist()
    try:
        contig_id = contigs.index(normalized.contig)
    except ValueError as error:
        raise KeyError(f"unknown contig {normalized.contig!r}") from error

    offsets = np.asarray(ds["offsets"].values)
    flat_start = int(offsets[contig_id])
    contig_length = int(offsets[contig_id + 1]) - flat_start
    start = 0 if normalized.start is None else normalized.start
    stop = contig_length if normalized.stop is None else normalized.stop

    if start < 0:
        raise ValueError("region start must be nonnegative")
    if stop <= start:
        raise ValueError("region must be nonempty with start less than stop")
    if stop > contig_length:
        raise ValueError(
            f"region stop {stop} exceeds contig length {contig_length}"
        )

    resolved = RegionQuery(normalized.contig, start, stop)
    return resolved, slice(flat_start + start, flat_start + stop)
