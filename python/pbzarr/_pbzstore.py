"""PbzStore: a container of tracks over the flat layout.

The store holds no genome and no region API — those live on `Track`.
"""
from __future__ import annotations

import zarr

from ._native import PbzError
from ._store import create_store as _create_store
from ._track import _DEFAULT_CHUNKS, Track


def _is_track(attrs: dict) -> bool:
    return any(c.get("name") == "perbase" for c in attrs.get("zarr_conventions", []))


# Read backend rides the handle, expressed in zarr's vocabulary: chunks=None -> eager
# numpy; chunks={} (or a dict / "auto") -> dask, aligned to the on-disk chunk grid.
# The store carries a default that flows to the tracks it produces (a track may
# override it via store.track(name, chunks=...)). Default is lazy/dask.
_UNSET = object()


class PbzStore:
    def __init__(self, path: str, *, chunks=_DEFAULT_CHUNKS):
        self.path = str(path)
        self.chunks = chunks
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
    def create(cls, path: str, *, chunks=_DEFAULT_CHUNKS) -> "PbzStore":
        _create_store(path)
        return cls(path, chunks=chunks)

    @classmethod
    def open(cls, path: str, *, chunks=_DEFAULT_CHUNKS) -> "PbzStore":
        return cls(path, chunks=chunks)

    def tracks(self) -> list[str]:
        root = zarr.open_group(self.path, mode="r")
        return sorted(n for n, node in root.members() if _is_track(dict(node.attrs)))

    def track(self, name: str, *, chunks=_UNSET) -> Track:
        return Track(self.path, name, chunks=self.chunks if chunks is _UNSET else chunks)

    def tree(self):
        import xarray as xr

        return xr.open_datatree(self.path, engine="zarr", consolidated=False)

    def import_bed_multi(
        self,
        bed: str,
        columns: dict[str, str],
        *,
        genome: str,
        workers: int | None = None,
        chunk_size: int | None = None,
        shard_size: int | None = None,
        progress: bool = False,
    ) -> list[Track]:
        """Import many BED columns into per-column scalar tracks in one pass.

        `columns` maps a header column name to a dtype ("int32", "float32",
        "bool", ...); each becomes a track named after the column. `genome` is a
        .fai / chrom.sizes path. Returns the created `Track` handles in order.
        """
        from ._native import import_bed_multi as _native_import_bed_multi

        items = list(columns.items())
        _native_import_bed_multi(
            self.path, str(bed), items, str(genome), workers, chunk_size, shard_size, progress
        )
        return [self.track(name) for name, _ in items]
