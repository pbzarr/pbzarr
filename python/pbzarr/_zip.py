"""Read-only access to single-file zip stores."""

from __future__ import annotations

import os

from zarr.storage import ZipStore

WRITE_ONCE = (
    "zip stores are write-once; this call cannot write to {path}. "
    "Use pbz import --zip to make a zip store"
)


def zip_path(source) -> str | None:
    """The local path of ``source`` when it names a zip store, else None.

    A local path is a zip store when it ends in ``.zip`` or is a regular file.
    """
    if not isinstance(source, (str, os.PathLike)):
        return None
    path = os.fspath(source)
    if not isinstance(path, str) or "://" in path:
        return None
    if path.lower().endswith(".zip") or os.path.isfile(path):
        return path
    return None


def refuse(source) -> None:
    path = zip_path(source)
    if path is not None:
        raise ValueError(WRITE_ONCE.format(path=path))


def open_store(path: str) -> ZipStore:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")
    return ZipStore(path, mode="r")
