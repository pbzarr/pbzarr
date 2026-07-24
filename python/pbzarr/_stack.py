"""Batch stack: combine single-sample .pbz stores into a cohort store."""
from __future__ import annotations

from ._pbzstore import PbzStore


def stack(
    sources,
    out: str,
    *,
    tracks=None,
    column_dim: str | None = None,
    column_chunk_size: int | None = None,
    workers: int | None = None,
    progress: bool = False,
) -> PbzStore:
    """Combine single-sample stores into a fresh cohort store at `out`.

    `sources` is a list of store paths, or `(path, label)` tuples; `label`
    defaults to the store's filename stem and becomes the sample's column label.
    Each scalar track shared by all sources becomes a `(ΣL, N)` cohort track.
    `tracks` selects a subset (default: every track of the first source). All
    sources must share a genome. `progress=True` shows a per-track progress bar
    on a terminal (periodic log lines otherwise). Returns the created cohort
    `PbzStore`.
    """
    norm: list[tuple[str, str | None]] = []
    for s in sources:
        if isinstance(s, (tuple, list)):
            norm.append((str(s[0]), s[1] if len(s) > 1 else None))
        else:
            norm.append((str(s), None))

    from ._native import stack as _native_stack

    _native_stack(
        norm,
        str(out),
        list(tracks) if tracks is not None else None,
        column_dim,
        column_chunk_size,
        workers,
        progress,
    )
    return PbzStore(str(out))
