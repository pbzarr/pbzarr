"""PbzStore: a container of tracks over the flat layout.

The store holds no genome and no region API — those live on `Track`.
"""
from __future__ import annotations

import zarr

from ._kind import is_perbase, kind_of
from ._native import PbzError
from ._store import create_store as _create_store
from ._track import _DEFAULT_CHUNKS, Track


def _is_track(attrs: dict) -> bool:
    return is_perbase(attrs) and kind_of(attrs) == "track"


def _track_meta(store_path: str, name: str) -> dict:
    return dict(zarr.open_group(f"{store_path}/{name}", mode="r").attrs)


def _wrap_values(zarr_arr, chunks):
    """Back the values array per the store's read config: chunks=None -> eager
    numpy; anything else -> dask aligned to the on-disk chunk grid."""
    if chunks is None:
        import numpy as np

        return np.asarray(zarr_arr[...])
    import dask.array as dask_array

    return dask_array.from_array(zarr_arr, chunks=zarr_arr.chunks)


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
        attrs = dict(root.attrs)
        if not is_perbase(attrs):
            raise PbzError(f"{self.path!r} is not a pbz store (no zarr_conventions marker)")
        if kind_of(attrs) == "track":
            raise PbzError(
                f"{self.path!r} is a pbz track, not a store; "
                "use Track.open(path) or pbzarr.open(path)"
            )

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

    def dataset(self, tracks: list[str] | None = None):
        """Assemble same-genome tracks into one `xr.Dataset` (the cross-track view).

        Every track must share a `genome_checksum` (raises `ValueError`
        otherwise). Variables share the flat `position` axis (no index); each
        cohort track's column labels ride on its column dim; the shared
        `contigs`/`offsets` and the genome identity travel as coords/attrs so
        the Dataset stays self-describing. Backing (numpy vs dask) follows the
        store's `chunks`.
        """
        import numpy as np
        import xarray as xr

        from ._read import open_track

        names = self.tracks() if tracks is None else list(tracks)
        if not names:
            raise PbzError("dataset(): store has no tracks")

        metas = {n: _track_meta(self.path, n) for n in names}
        checksums = {n: metas[n].get("perbase:genome_checksum") for n in names}
        if len(set(checksums.values())) > 1:
            listing = "\n".join(f"  {n}: {checksums[n]}" for n in names)
            raise ValueError(
                "dataset(): tracks do not share a genome (genome_checksum mismatch):\n"
                f"{listing}\n"
                "pass an explicit tracks=[...] subset that shares one genome"
            )

        data_vars = {}
        first = None
        for n in names:
            ta = open_track(self.path, n)
            if first is None:
                first = ta
            data = _wrap_values(ta.values, self.chunks)
            if ta.col_dim is None:
                da = xr.DataArray(data, dims=["position"])
            else:
                da = xr.DataArray(
                    data,
                    dims=["position", ta.col_dim],
                    coords={ta.col_dim: np.asarray(ta.labels)},
                )
                da.attrs["column_dim"] = ta.col_dim
            da.attrs["dtype"] = str(ta.values.dtype)
            fv = ta.values.fill_value
            if fv is not None:
                da.attrs["fill_value"] = fv.item() if hasattr(fv, "item") else fv
            for key in ("perbase:description", "perbase:source"):
                if metas[n].get(key) is not None:
                    da.attrs[key.split(":", 1)[1]] = metas[n][key]
            data_vars[n] = da

        coords = {
            "contigs": ("contig", np.asarray(first.contigs)),
            "offsets": ("contig_boundary", np.asarray(first.offsets)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        m0 = metas[names[0]]
        ds.attrs["genome_checksum"] = m0.get("perbase:genome_checksum")
        if m0.get("perbase:genome_name") is not None:
            ds.attrs["genome_name"] = m0["perbase:genome_name"]
        ds.attrs["coordinates"] = m0.get("perbase:coordinates", "0-based-half-open")
        # Let a downstream RegionView.to_pbz() find the on-disk source for the
        # Rust region-store builder.
        ds.attrs["pbz_source_path"] = self.path
        return ds

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
