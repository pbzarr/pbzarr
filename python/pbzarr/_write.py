"""Persistent, destination-oriented PBZ writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
import numbers
import os
from typing import Literal

import numpy as np

from . import _native, _zip
from ._types import Codecs, Fill, PathLike, Source


def _path(value: object, role: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as error:
        raise TypeError(f"{role} must be a string or path-like object") from error
    if not isinstance(path, str):
        raise TypeError(f"{role} must resolve to a string path")
    if not path:
        raise ValueError(f"{role} must not be empty")
    return path


def _is_pathlike(value: object) -> bool:
    try:
        return isinstance(os.fspath(value), str)
    except TypeError:
        return False


def _source(value: object) -> tuple[str, str | None]:
    if _is_pathlike(value):
        return (_path(value, "source path"), None)
    if (
        type(value) is tuple
        and len(value) == 2
        and _is_pathlike(value[0])
        and isinstance(value[1], str)
    ):
        return (_path(value[0], "source path"), value[1])
    raise TypeError("each source must be a path or an exact (path, label) tuple")


def _sources(values: Source | Iterable[Source]) -> list[tuple[str, str | None]]:
    """One source, or an iterable of them. A ``(path, str)`` tuple is always
    one labeled source; give a list for several sources."""
    if _is_pathlike(values) or (
        type(values) is tuple
        and len(values) == 2
        and _is_pathlike(values[0])
        and isinstance(values[1], str)
    ):
        normalized = [_source(values)]
    else:
        try:
            normalized = [_source(value) for value in values]
        except TypeError as error:
            if "each source" in str(error):
                raise
            raise TypeError("sources must be one source or an iterable of sources") from error
    if not normalized:
        raise ValueError("sources must not be empty")
    return normalized


def _codecs_json(codecs: Codecs | None) -> str | None:
    if codecs is None:
        return None
    if not isinstance(codecs, (list, dict)):
        raise TypeError(
            "codecs must be a list of codec metadata dicts or a"
            " {chunk_grid, codecs} mapping"
        )
    return json.dumps(codecs)


def _fill(value: object) -> str | None:
    """The fill as text for the native import: ``na``, ``true``, ``false``, or a number."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, str):
        if value == "na":
            return value
    elif isinstance(value, numbers.Integral):
        return str(int(value))
    elif isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"fill {value!r} is not finite; use 'na' for a missing value")
        return str(int(number)) if number.is_integer() else repr(number)
    raise ValueError(f"fill must be a number, True, False, or 'na'; got {value!r}")


def _destination(value: object) -> str:
    path = _path(value, "destination")
    if "://" in path:
        raise ValueError("import writes local stores only; give a local path")
    _zip.refuse(path)
    return path


def import_d4(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    dim: str | None = None,
    threads: int | None = None,
    decode_chunks: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    progress: bool = False,
    codecs: Codecs | None = None,
) -> dict[str, int | float]:
    """Import one or more D4 sources into a PBZ store.

    When the destination does not exist, the import creates it from the
    source headers. When it exists, the import adds the track to it. A
    track name that the store already uses is refused.
    `sources` is a path, a `(path, label)` tuple, or a list of them. Give
    labels for all sources or for none; without labels, each column takes
    the file name without its format extension. A missing source file
    raises `FileNotFoundError`, and a bad argument raises `ValueError`.
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _destination(destination)

    normalized_sources = _sources(sources)
    return _native.import_d4(
        destination=destination_path,
        track=track,
        sources=normalized_sources,
        dim=dim,
        threads=threads,
        chunk_size=chunk_size,
        column_chunk_size=column_chunk_size,
        shard_size=shard_size,
        shard_column_size=shard_column_size,
        progress=progress,
        codecs=_codecs_json(codecs),
        decode_chunks=decode_chunks,
    )


def import_bigwig(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    dim: str | None = None,
    fill: Fill | None = None,
    threads: int | None = None,
    decode_chunks: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    progress: bool = False,
    codecs: Codecs | None = None,
) -> dict[str, int | float]:
    """Import one or more bigWig sources into a PBZ store.

    The track is ``float32``. ``fill=`` is the value of bases that no bigWig
    interval covers: a number, or ``"na"``. ``None`` (the default) gives
    ``"na"``: such a base is missing and reads as NaN. Give ``fill=0`` when
    the gaps mean 0. A bad fill raises ``ValueError``.

    When the destination does not exist, the import creates it from the
    source headers. When it exists, the import adds the track to it. A
    track name that the store already uses is refused.
    `sources` is a path, a `(path, label)` tuple, or a list of them. Give
    labels for all sources or for none; without labels, each column takes
    the file name without its format extension. A missing source file
    raises `FileNotFoundError`, and a bad argument raises `ValueError`.
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _destination(destination)

    normalized_sources = _sources(sources)
    return _native.import_bigwig(
        destination=destination_path,
        track=track,
        sources=normalized_sources,
        dim=dim,
        fill=_fill(fill),
        threads=threads,
        chunk_size=chunk_size,
        column_chunk_size=column_chunk_size,
        shard_size=shard_size,
        shard_column_size=shard_column_size,
        progress=progress,
        codecs=_codecs_json(codecs),
        decode_chunks=decode_chunks,
    )


def _bed_selector(value: object) -> str:
    """Normalize a BED column selector: a header name, or a 1-based number."""
    if isinstance(value, bool):
        raise TypeError("BED column selectors must be header names or 1-based integers")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"BED column selectors are 1-based; got {value}")
        return str(value)
    if isinstance(value, str):
        if not value:
            raise ValueError("BED column selector must not be empty")
        return value
    raise TypeError("BED column selectors must be header names or 1-based integers")


_SCHEMA_KEYS = frozenset({"track", "dtype", "fill"})


def _schema_entry(key: object, value: object) -> tuple[str, str | None, str | None, str | None]:
    selector = _bed_selector(key)
    if value is None or isinstance(value, str):
        return (selector, None, value, None)
    if not isinstance(value, Mapping):
        raise TypeError("schema entries must be dtype strings, None, or dicts")
    unknown = set(value) - _SCHEMA_KEYS
    if unknown:
        raise ValueError(
            f"schema entry {key!r} has unknown keys {sorted(map(str, unknown))}; "
            "use track, dtype, and fill"
        )
    track, dtype = value.get("track"), value.get("dtype")
    if not all(item is None or isinstance(item, str) for item in (track, dtype)):
        raise TypeError(f"schema entry {key!r}: track and dtype must be strings or None")
    return (selector, track, dtype, _fill(value.get("fill")))


def import_bed(
    destination: PathLike,
    sources: Source | Iterable[Source],
    *,
    genome: PathLike | None = None,
    schema: Mapping[str | int, str | Mapping[str, object] | None] | None = None,
    track: str | None = None,
    column: str | int | None = None,
    dtype: str | None = None,
    fill: Fill | None = None,
    dim: str | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    codecs: Codecs | None = None,
    threads: int | None = None,
    decode_chunks: int | None = None,
    progress: bool = False,
) -> dict[str, int | float]:
    """Import tabix-indexed BED sources into a PBZ store.

    The call has three forms, matching `pbz import bed`:

    - `column=` selects one BED column (a header name, or a 1-based column
      number of 4 or more) and imports it as one track. `track=` names the
      track; it defaults to the header name, so a numbered column needs
      `track=`. `dtype=` skips inference.
    - `schema=` maps BED columns (header names, or 1-based column numbers)
      to entries and imports one track per entry. An entry is a dtype
      string, `None`, or a dict with the optional keys `"track"`, `"dtype"`,
      and `"fill"`. The track takes the header name when the entry gives
      none, so a numbered column needs `"track"`. A missing dtype is
      inferred across every source, and a missing fill takes `fill=`.
      `schema=` conflicts with `track=`, `column=`, and `dtype=`.
    - With none of those, BED3 input imports a bool presence track and
      headerless BED4 one value track (both need `track=`). Headered input
      imports every value column: one wide track when a single source has
      `track=` set, else one track per column.

    Output is 2D when there are several sources, or when one track holds
    several BED columns. Every 2D output requires `dim=`.

    `fill=` is the value of bases that no interval covers in a value
    column: a number, `True` or `False` for a bool column, or `"na"`.
    `"na"` is the null of the dtype: NaN for a float, the type maximum for
    an integer. `None` (the default) gives `"na"`, and `False` for a bool
    column. With `"na"`, dtype inference keeps the type maximum free. Give
    `fill=0` when the gaps mean 0. A BED3 mask always holds `False` and
    refuses `fill=`. A bad fill raises `ValueError`.

    xarray decodes an integer track with the null to a float dtype, with
    NaN at the nulls: `float32` for 8- and 16-bit integers, `float64` for
    32- and 64-bit integers.

    BED files have no contig lengths. When the destination does not exist,
    the import creates it with the lengths in `genome=`, a .fai or
    chrom.sizes file. When it exists, the store gives the lengths, and
    `genome=` is optional; if given, it must match the store. A track name
    that the store already uses is refused.
    `sources` is a path, a `(path, label)` tuple, or a list of them. Give
    labels for all sources or for none; without labels, each column takes
    the file name without its format extension. A missing source file
    raises `FileNotFoundError`, and a bad argument raises `ValueError`.
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _destination(destination)

    normalized_sources = _sources(sources)
    genome_path = _path(genome, "genome") if genome is not None else None
    schema_entries = None
    if schema is not None:
        if track is not None or column is not None or dtype is not None:
            raise ValueError("schema= conflicts with track=, column=, and dtype=")
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping of BED column selectors to entries")
        if not schema:
            raise ValueError("schema must not be empty")
        schema_entries = [_schema_entry(key, value) for key, value in schema.items()]
    normalized_column = _bed_selector(column) if column is not None else None
    return _native.import_bed(
        destination=destination_path,
        sources=normalized_sources,
        genome=genome_path,
        schema=schema_entries,
        track=track,
        column=normalized_column,
        dtype=dtype,
        fill=_fill(fill),
        dim=dim,
        threads=threads,
        chunk_size=chunk_size,
        column_chunk_size=column_chunk_size,
        shard_size=shard_size,
        shard_column_size=shard_column_size,
        progress=progress,
        codecs=_codecs_json(codecs),
        decode_chunks=decode_chunks,
    )


def import_bam(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    mode: Literal["depth", "composition"] = "depth",
    fasta: PathLike | None = None,
    min_mapq: int = 0,
    exclude_flags: int = 1796,
    min_bq: int = 0,
    overlap: Literal["proper", "all", "none"] = "proper",
    count_deletions: bool = False,
    dim: str | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    threads: int | None = None,
    decode_chunks: int | None = None,
    progress: bool = False,
    codecs: Codecs | None = None,
) -> dict[str, int | float]:
    """Import per-base depth or composition counts from BAM/CRAM sources.

    `mode` is `"depth"` (writes `track`) or `"composition"` (writes `track`
    plus one `track_{field}` per composition field). CRAM sources need
    `fasta`; BAM sources ignore it. `count_deletions` defaults to
    `False`, matching the samtools/mosdepth/Picard/riker convention that
    CIGAR `D` positions do not count toward depth; set `True` to count
    deletion spans toward depth (useful for callability masks). `overlap`
    controls mate-overlap dedup: `"proper"` (default) matches mosdepth,
    collapsing only PROPER_PAIR-flagged overlapping mates to one count;
    `"all"` matches riker/samtools-mpileup-style unconditional dedup of any
    overlapping mate pair; `"none"` disables dedup, double-counting every
    overlapping pair's shared span. When the destination does not exist,
    the import creates it from the source headers. When it exists, the
    import adds the track(s) to it. A track name that the store already
    uses is refused. `decode_chunks=` sets the position chunks each reader
    decodes per pass (default auto). Returns the import report dict.

    `sources` is a path, a `(path, label)` tuple, or a list of them. Give
    labels for all sources or for none; without labels, each column takes
    the file name without its format extension. A missing source file
    raises `FileNotFoundError`, and a bad argument raises `ValueError`.
    """
    destination_path = _destination(destination)

    if overlap not in ("proper", "all", "none"):
        raise ValueError('overlap must be "proper", "all", or "none"')
    normalized_sources = _sources(sources)
    fasta_path = _path(fasta, "fasta") if fasta is not None else None
    return _native.import_bam(
        destination=destination_path,
        track=track,
        sources=normalized_sources,
        mode=mode,
        fasta=fasta_path,
        min_mapq=min_mapq,
        exclude_flags=exclude_flags,
        min_bq=min_bq,
        overlap=overlap,
        count_deletions=count_deletions,
        dim=dim,
        threads=threads,
        chunk_size=chunk_size,
        column_chunk_size=column_chunk_size,
        shard_size=shard_size,
        shard_column_size=shard_column_size,
        progress=progress,
        codecs=_codecs_json(codecs),
        decode_chunks=decode_chunks,
    )
