"""Type stubs for the PyO3 extension."""
from typing import Container, Sequence

class PbzError(RuntimeError): ...

def import_d4(
    *,
    destination: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    dim: str | None = ...,
    threads: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    decode_chunks: int | None = ...,
) -> dict[str, int | float]: ...
def import_bigwig(
    *,
    destination: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    dim: str | None = ...,
    threads: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    decode_chunks: int | None = ...,
) -> dict[str, int | float]: ...
def import_bed(
    *,
    destination: str,
    sources: Sequence[tuple[str, str | None]],
    genome: str | None = ...,
    schema: Sequence[tuple[str, str | None]] | None = ...,
    track: str | None = ...,
    column: str | None = ...,
    dtype: str | None = ...,
    dim: str | None = ...,
    threads: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    decode_chunks: int | None = ...,
) -> dict[str, int | float]: ...
def import_bam(
    *,
    destination: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    mode: str = ...,
    fasta: str | None = ...,
    min_mapq: int = ...,
    exclude_flags: int = ...,
    min_bq: int = ...,
    overlap: str = ...,
    count_deletions: bool = ...,
    dim: str | None = ...,
    threads: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    shard_size: int | None = ...,
    shard_column_size: int | None = ...,
    progress: bool = ...,
    codecs: str | None = ...,
    decode_chunks: int | None = ...,
) -> dict[str, int | float]: ...
def create_dataset(*, destination: str, reference: Sequence[tuple[str, int]]) -> None: ...
def parse_region(
    *, text: str, contigs: Container[str] | None = ...
) -> tuple[str, int | None, int | None]: ...
