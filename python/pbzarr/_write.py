"""Persistent, destination-oriented PBZ writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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


def _sources(values: Source | Iterable[Source]) -> list[tuple[str, str | None]]:
    if _is_pathlike(values):
        normalized = [_source(values)]
    elif (
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


def import_d4(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    column_dim: str | None = None,
    workers: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    progress: bool = False,
) -> None:
    """Import one or more D4 sources into an existing PBZ collection."""
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
        progress,
    )


def import_bigwig(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    column_dim: str | None = None,
    workers: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    progress: bool = False,
) -> None:
    """Import one or more bigWig sources into an existing PBZ collection."""
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
        progress,
    )


def import_bed(
    destination: PathLike,
    track: str,
    sources: Source | Iterable[Source],
    *,
    column: str,
    dtype: str,
    genome: PathLike,
    column_dim: str | None = None,
    workers: int | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    progress: bool = False,
) -> None:
    """Import one column from one or more BED sources into a PBZ collection."""
    destination_path = _path(destination, "destination")
    normalized_sources = _sources(sources)
    genome_path = _path(genome, "genome")
    _require_collection(destination_path)
    return _native.import_bed(
        destination_path,
        track,
        normalized_sources,
        column,
        dtype,
        genome_path,
        column_dim,
        workers,
        chunk_size,
        column_chunk_size,
        progress,
    )


def import_bed_multi(
    destination: PathLike,
    bed: PathLike,
    columns: Mapping[str, str],
    *,
    genome: PathLike,
    workers: int | None = None,
    chunk_size: int | None = None,
    shard_size: int | None = None,
    progress: bool = False,
) -> None:
    """Import an ordered mapping of BED columns as scalar tracks."""
    destination_path = _path(destination, "destination")
    bed_path = _path(bed, "bed")
    genome_path = _path(genome, "genome")
    if not isinstance(columns, Mapping):
        raise TypeError("columns must be a mapping of track names to dtypes")
    items = list(columns.items())
    if not items:
        raise ValueError("columns must not be empty")
    if any(not isinstance(name, str) or not isinstance(dtype, str) for name, dtype in items):
        raise TypeError("column names and dtypes must be strings")
    _require_collection(destination_path)
    return _native.import_bed_multi(
        destination_path,
        bed_path,
        items,
        genome_path,
        workers,
        chunk_size,
        shard_size,
        progress,
    )


def stack(
    sources: Source | Iterable[Source],
    destination: PathLike,
    *,
    tracks: Sequence[str] | None = None,
    column_dim: str | None = None,
    column_chunk_size: int | None = None,
    workers: int | None = None,
) -> None:
    """Stack scalar tracks from PBZ collections into a new cohort collection."""
    normalized_sources = _sources(sources)
    destination_path = _path(destination, "destination")
    _require_absent(destination_path)
    for source_path, _ in normalized_sources:
        _require_collection(source_path)
    normalized_tracks = list(tracks) if tracks is not None else None
    return _native.stack(
        normalized_sources,
        destination_path,
        normalized_tracks,
        column_dim,
        column_chunk_size,
        workers,
    )
