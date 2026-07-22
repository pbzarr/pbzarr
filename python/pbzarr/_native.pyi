"""Type stubs for the PyO3 extension."""
from typing import Sequence

class PbzError(RuntimeError): ...

def import_d4(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def import_bigwig(
    store_path: str,
    track: str,
    sources: Sequence[tuple[str, str | None]],
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
    workers: int | None = ...,
    chunk_size: int | None = ...,
    column_chunk_size: int | None = ...,
    progress: bool = ...,
) -> None: ...
def d4_contigs(path: str) -> list[tuple[str, int]]: ...
def bigwig_contigs(path: str) -> list[tuple[str, int]]: ...
