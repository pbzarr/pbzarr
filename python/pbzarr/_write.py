"""Persistent, destination-oriented PBZ writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from typing import TypeAlias

from . import _native
from ._native import PbzError
from ._open import _read_root_attrs
from ._xarray import _node_kind

PathLike: TypeAlias = str | os.PathLike[str]
Source: TypeAlias = PathLike | tuple[PathLike, str]


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
    raise TypeError("each source must be a path or an exact (path, column_label) tuple")


_SOURCE_SUFFIXES = (".d4", ".bw", ".bigwig", ".bed", ".bed.gz", ".bam", ".cram")


def _sources(values: Source | Iterable[Source]) -> list[tuple[str, str | None]]:
    if _is_pathlike(values):
        normalized = [_source(values)]
    elif (
        type(values) is tuple
        and len(values) == 2
        and _is_pathlike(values[0])
        and isinstance(values[1], str)
    ):
        if values[1].lower().endswith(_SOURCE_SUFFIXES):
            raise ValueError(
                f"source label {values[1]!r} looks like a source file; pass a"
                " list to import multiple sources, or rename the label"
            )
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


def _codecs_json(codecs: list | dict | None) -> str | None:
    if codecs is None:
        return None
    if not isinstance(codecs, (list, dict)):
        raise TypeError(
            "codecs must be a list of codec metadata dicts or a"
            " {chunk_grid, codecs} mapping"
        )
    return json.dumps(codecs)


def _require_absent(path: str) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"destination already exists: {path}")


def _require_collection(path: str) -> None:
    if not os.path.isdir(path):
        raise PbzError(f"destination is not an existing PBZ collection: {path}")
    try:
        attrs = _read_root_attrs(path, storage_options=None)
        kind = _node_kind(attrs)
    except PbzError:
        raise
    except Exception as error:
        raise PbzError(f"destination is not a PBZ collection: {path}") from error
    if kind != "collection":
        raise PbzError(f"destination is not a PBZ collection: {path}")


def create_store(destination: PathLike, /) -> None:
    """Create an empty PBZ collection at an absent filesystem destination."""
    destination_path = _path(destination, "destination")
    _require_absent(destination_path)
    return _native.create_store(destination_path)


def _scales(scales: Sequence[int] | None) -> list[int] | None:
    if scales is None:
        return None
    return [int(factor) for factor in scales]


def import_d4(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    column_dim: str | None = None,
    workers: int | None = None,
    in_flight: int | None = None,
    decode_chunks: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    progress: bool = False,
    codecs: list | dict | None = None,
    scales: Sequence[int] | None = None,
) -> dict[str, int | float]:
    """Import one or more D4 sources into an existing PBZ collection.

    `scales=` builds a multiscale pyramid on the new track with the given
    downsampling factors; `None` or empty builds no pyramid. `in_flight=`
    caps the position spans the engine holds open at once (`None` = auto).
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _path(destination, "destination")
    normalized_sources = _sources(sources)
    _require_collection(destination_path)
    return _native.import_d4(
        destination_path,
        track,
        normalized_sources,
        column_dim,
        workers,
        chunk_size,
        column_chunk_size,
        shard_size,
        shard_column_size,
        progress,
        codecs=_codecs_json(codecs),
        scales=_scales(scales),
        in_flight=in_flight,
        decode_chunks=decode_chunks,
    )


def import_bigwig(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    column_dim: str | None = None,
    workers: int | None = None,
    in_flight: int | None = None,
    decode_chunks: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    progress: bool = False,
    codecs: list | dict | None = None,
    scales: Sequence[int] | None = None,
) -> dict[str, int | float]:
    """Import one or more bigWig sources into an existing PBZ collection.

    `scales=` builds a multiscale pyramid on the new track with the given
    downsampling factors; `None` or empty builds no pyramid. `in_flight=`
    caps the position spans the engine holds open at once (`None` = auto).
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _path(destination, "destination")
    normalized_sources = _sources(sources)
    _require_collection(destination_path)
    return _native.import_bigwig(
        destination_path,
        track,
        normalized_sources,
        column_dim,
        workers,
        chunk_size,
        column_chunk_size,
        shard_size,
        shard_column_size,
        progress,
        codecs=_codecs_json(codecs),
        scales=_scales(scales),
        in_flight=in_flight,
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


def import_bed(
    destination: PathLike,
    sources: Source | Iterable[Source],
    *,
    genome: PathLike,
    schema: Mapping[str | int, str | None] | None = None,
    track: str | None = None,
    column: str | int | None = None,
    dtype: str | None = None,
    column_dim: str | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    codecs: list | dict | None = None,
    workers: int | None = None,
    in_flight: int | None = None,
    decode_chunks: int | None = None,
    progress: bool = False,
    scales: Sequence[int] | None = None,
) -> dict[str, int | float]:
    """Import tabix-indexed BED sources into an existing PBZ collection.

    The call has three forms, matching `pbz import bed`:

    - `column=` selects one BED column (a header name, or a 1-based column
      number) and imports it as one track. `track=` names the track;
      it defaults to the column name. `dtype=` skips inference.
    - `schema=` maps BED column selectors to dtype strings and imports one
      track per entry, named by its selector. A `None` dtype is inferred
      across every source. `schema=` conflicts with `track=`, `column=`,
      and `dtype=`.
    - With none of those, BED3 input imports a bool presence track and
      headerless BED4 one value track (both need `track=`). Headered input
      imports every value column: one wide track when a single source has
      `track=` set, else one track per column.

    Output is 2D when there are several sources, or when one track holds
    several BED columns. Every 2D output requires `column_dim=`.
    `scales=` builds a multiscale pyramid on each new track with the given
    downsampling factors; `None` or empty builds no pyramid. `in_flight=`
    caps the position spans the engine holds open at once (`None` = auto).
    `decode_chunks=` sets the position chunks each reader decodes per pass
    (default auto). Returns the import report dict.
    """
    destination_path = _path(destination, "destination")
    normalized_sources = _sources(sources)
    genome_path = _path(genome, "genome")
    schema_entries = None
    if schema is not None:
        if track is not None or column is not None or dtype is not None:
            raise ValueError("schema= conflicts with track=, column=, and dtype=")
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a mapping of BED column selectors to dtypes")
        if not schema:
            raise ValueError("schema must not be empty")
        schema_entries = []
        for key, value in schema.items():
            if value is not None and not isinstance(value, str):
                raise TypeError("schema dtypes must be strings or None")
            schema_entries.append((_bed_selector(key), value))
    normalized_column = _bed_selector(column) if column is not None else None
    _require_collection(destination_path)
    return _native.import_bed(
        destination_path,
        normalized_sources,
        genome_path,
        schema_entries,
        track,
        normalized_column,
        dtype,
        column_dim,
        workers,
        chunk_size,
        column_chunk_size,
        shard_size,
        shard_column_size,
        progress,
        codecs=_codecs_json(codecs),
        scales=_scales(scales),
        in_flight=in_flight,
        decode_chunks=decode_chunks,
    )


def import_bam(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    mode: str = "depth",
    reference: PathLike | None = None,
    min_mapq: int = 0,
    exclude_flags: int = 1796,
    min_bq: int = 0,
    overlap: str = "proper",
    count_deletions: bool = False,
    column_labels: Sequence[str] | None = None,
    column_dim: str | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    workers: int | None = None,
    in_flight: int | None = None,
    decode_chunks: int | None = None,
    progress: bool = False,
    codecs: list | dict | None = None,
    scales: Sequence[int] | None = None,
) -> dict[str, int | float]:
    """Import per-base depth or composition counts from BAM/CRAM sources.

    `mode` is `"depth"` (writes `track`) or `"composition"` (writes `track`
    plus one `track_{field}` per composition field). CRAM sources need
    `reference`; BAM sources ignore it. `count_deletions` defaults to
    `False`, matching the samtools/mosdepth/Picard/riker convention that
    CIGAR `D` positions do not count toward depth; set `True` to count
    deletion spans toward depth (useful for callability masks). `overlap`
    controls mate-overlap dedup: `"proper"` (default) matches mosdepth,
    collapsing only PROPER_PAIR-flagged overlapping mates to one count;
    `"all"` matches riker/samtools-mpileup-style unconditional dedup of any
    overlapping mate pair; `"none"` disables dedup, double-counting every
    overlapping pair's shared span. `scales=` builds a multiscale pyramid on
    each new track with the given downsampling factors; `None` or empty
    builds no pyramid. `in_flight=` caps the position spans the engine holds
    open at once (`None` = auto). `decode_chunks=` sets the position chunks
    each reader decodes per pass (default auto). Returns the import report
    dict.
    """
    destination_path = _path(destination, "destination")
    if overlap not in ("proper", "all", "none"):
        raise ValueError('overlap must be "proper", "all", or "none"')
    normalized_sources = _sources(sources)
    if column_labels is not None:
        labels = list(column_labels)
        if len(labels) != len(normalized_sources):
            raise ValueError("column_labels must match the number of sources")
        merged: list[tuple[str, str | None]] = []
        for (path, existing), label in zip(normalized_sources, labels):
            if existing is not None:
                raise ValueError(
                    "column_labels conflicts with a (path, label) source tuple"
                )
            merged.append((path, label))
        normalized_sources = merged
    reference_path = _path(reference, "reference") if reference is not None else None
    _require_collection(destination_path)
    return _native.import_bam(
        destination_path,
        track,
        normalized_sources,
        mode,
        reference_path,
        min_mapq,
        exclude_flags,
        min_bq,
        overlap,
        count_deletions,
        column_dim,
        workers,
        chunk_size,
        column_chunk_size,
        shard_size,
        shard_column_size,
        progress,
        codecs=_codecs_json(codecs),
        scales=_scales(scales),
        in_flight=in_flight,
        decode_chunks=decode_chunks,
    )
