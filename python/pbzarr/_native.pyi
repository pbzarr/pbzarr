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
) -> None: ...
