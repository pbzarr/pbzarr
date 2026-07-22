"""create_store: make an empty flat pbz store (bare marker root)."""
from __future__ import annotations


def create_store(path: str) -> None:
    """Create a new empty flat pbz store at `path`. Add tracks by importing."""
    from ._native import create_store as _native_create_store

    _native_create_store(str(path))
