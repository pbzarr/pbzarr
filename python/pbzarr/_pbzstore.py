"""PbzStore: a container of tracks over the flat layout.

The store holds no genome and no region API — those live on `Track`.
"""
from __future__ import annotations

import zarr

from ._native import PbzError
from ._store import create_store as _create_store
from ._track import Track


def _is_track(attrs: dict) -> bool:
    return any(c.get("name") == "perbase" for c in attrs.get("zarr_conventions", []))


class PbzStore:
    def __init__(self, path: str):
        self.path = str(path)
        try:
            root = zarr.open_group(self.path, mode="r")
        except Exception as e:  # noqa: BLE001 - surface as PbzError
            raise PbzError(f"cannot open pbz store {self.path!r}: {e}") from e
        if not any(
            c.get("name") == "perbase"
            for c in dict(root.attrs).get("zarr_conventions", [])
        ):
            raise PbzError(f"{self.path!r} is not a pbz store (no zarr_conventions marker)")

    @classmethod
    def create(cls, path: str) -> "PbzStore":
        _create_store(path)
        return cls(path)

    def tracks(self) -> list[str]:
        root = zarr.open_group(self.path, mode="r")
        return sorted(n for n, node in root.members() if _is_track(dict(node.attrs)))

    def track(self, name: str) -> Track:
        return Track(self.path, name)

    def tree(self):
        import xarray as xr

        return xr.open_datatree(self.path, engine="zarr", consolidated=False)
