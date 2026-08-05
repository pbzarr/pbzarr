"""Type stubs for the PyO3 extension."""
from typing import Sequence

class PbzError(RuntimeError): ...

def import_d4(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def import_bigwig(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def import_bed(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    column: str,
    dtype: str,
    genome: str,
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def import_bed_multi(
    store_path: str,
    bed_gz: str,
    columns: Sequence[tuple[str, str]],
    genome: str,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    shard_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def create_store(store_path: str) -> None: ...
def stack(
    sources: Sequence[tuple[str, str | None]],
    out: str,
    tracks: Sequence[str] | None = ...,
    column_dim: str | None = ...,
    column_chunk_size: int | None = ...,
    workers: int | None = ...,
) -> None: ...
def d4_contigs(path: str) -> list[tuple[str, int]]: ...
def bigwig_contigs(path: str) -> list[tuple[str, int]]: ...
