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
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    scales: Sequence[int] | None = ...,
) -> dict[str, int]: ...
def import_bigwig(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    scales: Sequence[int] | None = ...,
) -> dict[str, int]: ...
def import_bed(
    store_path: str,
    sources: Sequence[tuple[str, str | None]],
    genome: str,
    schema: Sequence[tuple[str, str | None]] | None = ...,
    track: str | None = ...,
    column: str | None = ...,
    dtype: str | None = ...,
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    scales: Sequence[int] | None = ...,
) -> dict[str, int]: ...
def import_bam(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    mode: str = ...,
    reference: str | None = ...,
    min_mapq: int = ...,
    exclude_flags: int = ...,
    min_bq: int = ...,
    overlap: str = ...,
    count_deletions: bool = ...,
    column_dim: str | None = ...,
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    scales: Sequence[int] | None = ...,
) -> dict[str, int]: ...
def create_store(store_path: str) -> None: ...
