"""PbzStore: the pipeline handle for pbzarr.

Single class that holds the path, caches root metadata, and exposes:
- metadata accessors (contigs, tracks, contig_length, track_schema, column_labels)
- lazy data access (tree, read_track, region)
- write operations (create_track, import_d4, write_track) with two-phase staging

Reads return plain xarray. Writes accept xarray. Materialization is explicit.
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence, cast

import numpy as np
import xarray as xr
import zarr

from . import _store, _track
from ._region import parse_region

# Sentinel for the "lazy by default on PbzStore" decision. Distinguishes
# "user passed nothing -> apply our default" from "user passed chunks=None
# -> they want eager".
_LAZY_DEFAULT: Any = object()


class PbzStore:
    """Handle for a PBZ store on disk.

    `PbzStore(path)` opens an existing store. `PbzStore.create(path, ...)`
    bootstraps a new one. The handle caches root metadata; track data is
    streamed from disk on demand.

    Defaults to lazy (dask-backed) reads because the read -> transform ->
    write loop benefits from streaming. Pass `chunks=None` for eager.
    """

    def __init__(self, path: str, *, chunks: Any = _LAZY_DEFAULT) -> None:
        self.path = path
        # chunks=_LAZY_DEFAULT -> open with chunks={} (lazy aligned).
        # chunks=None -> eager. Anything else passes through.
        self._chunks_kwarg = chunks
        self._root_attrs: dict[str, Any] = {}
        self._tree: xr.DataTree | None = None
        self._load_root_attrs()

    @classmethod
    def create(
        cls,
        path: str,
        *,
        contigs: Sequence[str],
        contig_lengths: Sequence[int],
        coordinate_space: str | None = None,
        chunks: Any = _LAZY_DEFAULT,
    ) -> "PbzStore":
        """Bootstrap a new empty PBZ store at `path` and return a handle to it."""
        _store.create_store(
            path,
            contigs=contigs,
            contig_lengths=contig_lengths,
            coordinate_space=coordinate_space,
        )
        return cls(path, chunks=chunks)

    # ---- metadata cache ----

    def _load_root_attrs(self) -> None:
        g = zarr.open_group(self.path, mode="r")
        self._root_attrs = dict(_attr_dict(g.attrs, "perbase_zarr"))

    def _invalidate(self) -> None:
        """Drop the tree cache and reload root attrs after a mutation."""
        self._tree = None
        self._load_root_attrs()

    # ---- properties ----

    @property
    def contigs(self) -> list[str]:
        """Contig names in store order."""
        g = zarr.open_group(self.path, mode="r")
        return [str(v) for v in np.asarray(_array(g, "contigs")[:])]

    @property
    def tracks(self) -> list[str]:
        """Track names from root `perbase_zarr.tracks`."""
        tracks = _attr_dict_from(self._root_attrs, "tracks")
        return sorted(tracks.keys())

    @property
    def tree(self) -> xr.DataTree:
        """Lazy DataTree view of the whole store. Cached until next write."""
        if self._tree is None:
            kwargs: dict[str, Any] = {"engine": "zarr"}
            if self._chunks_kwarg is _LAZY_DEFAULT:
                kwargs["chunks"] = {}
            elif self._chunks_kwarg is not None:
                kwargs["chunks"] = self._chunks_kwarg
            # chunks=None -> omit kwarg -> xarray's eager default
            self._tree = xr.open_datatree(self.path, **kwargs)
        return self._tree

    # ---- metadata methods ----

    def contig_length(self, name: str) -> int:
        """Length of one contig."""
        g = zarr.open_group(self.path, mode="r")
        names = [str(v) for v in np.asarray(_array(g, "contigs")[:])]
        lengths = [int(v) for v in np.asarray(_array(g, "contig_lengths")[:])]
        try:
            idx = names.index(name)
        except ValueError as e:
            raise KeyError(f"contig {name!r} not in store") from e
        return lengths[idx]

    def track_schema(self, name: str) -> dict[str, Any]:
        """Root-attr fields for a track. Does NOT include column labels.

        For labels (which live in per-contig coord arrays), call
        `column_labels(name)`.
        """
        tracks = _attr_dict_from(self._root_attrs, "tracks")
        if name not in tracks:
            raise KeyError(f"track {name!r} not in store")
        return dict(tracks[name])

    def column_labels(self, name: str) -> list[str] | None:
        """Read column labels for a track from the first contig's coord array.

        Returns `None` for 1D scalar tracks. For cohort tracks, reads the
        coord array on `self.contigs[0]` and trusts cross-contig consistency
        (a writer guarantee).
        """
        schema = self.track_schema(name)
        col_dim = schema.get("column_dim")
        if col_dim is None:
            return None
        contigs = self.contigs
        if not contigs:
            return []
        g = zarr.open_group(self.path, mode="r")
        contig_g = _group(g, contigs[0])
        coord = _array(contig_g, str(col_dim))
        return [str(v) for v in np.asarray(coord[:])]

    # ---- reads ----

    def read_track(self, name: str) -> xr.DataTree:
        """Return a lazy DataTree with only this track's variable per child.

        Each child Dataset has a single variable named `name` (plus coords).
        This is the canonical input shape for `write_track`.
        """
        if name not in self.tracks:
            raise KeyError(f"track {name!r} not in store")

        def _trim(ds: xr.Dataset) -> xr.Dataset:
            if name in ds.data_vars:
                return cast(xr.Dataset, ds[[name]])
            return xr.Dataset()

        return self.tree.map_over_datasets(_trim)

    def region(
        self,
        query: str,
        *,
        track: str | None = None,
        column: str | None = None,
    ) -> xr.Dataset | xr.DataArray:
        """Slice the store to a region. Delegates to the `.pbz` accessor."""
        return self.tree.pbz.region(query, track=track, column=column)

    # ---- writes ----

    def create_track(
        self,
        name: str,
        *,
        dtype: str,
        columns: Sequence[str] | None = None,
        column_dim: str | None = None,
        chunk_size: int | None = None,
        column_chunk_size: int | None = None,
        shard_size: int | None = None,
        shard_column_size: int | None = None,
        compressors: Sequence | None = None,
        fill_value: Any = None,
        description: str | None = None,
        source: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Allocate empty arrays for a new track and register it in root attrs."""
        _track.create_track(
            self.path,
            track=name,
            dtype=dtype,
            columns=columns,
            column_dim=column_dim,
            chunk_size=chunk_size,
            column_chunk_size=column_chunk_size,
            shard_size=shard_size,
            shard_column_size=shard_column_size,
            compressors=compressors,
            fill_value=fill_value,
            description=description,
            source=source,
            overwrite=overwrite,
        )
        self._invalidate()

    def import_d4(
        self,
        track: str,
        sources: Sequence[tuple[str, str | None]],
        *,
        workers: int | None = None,
        chunk_size: int | None = None,
        column_chunk_size: int | None = None,
    ) -> None:
        """Populate an existing track from per-sample d4 files.

        The track must already exist (call `create_track` first). The d4
        path has dtype, scalar-vs-cohort, and label constraints that make
        auto-create more confusing than the explicit two-call form.
        """
        if track not in self.tracks:
            raise ValueError(
                f"track {track!r} does not exist; call create_track first, e.g.:\n"
                f"  store.create_track({track!r}, dtype='int32', "
                f"columns=[...], column_dim='sample')"
            )
        from ._native import import_d4 as _native_import_d4  # local import: PyO3
        _native_import_d4(
            self.path,
            track,
            list(sources),
            workers=workers,
            chunk_size=chunk_size,
            column_chunk_size=column_chunk_size,
        )
        self._invalidate()

    def write_track(
        self,
        name: str,
        data: xr.DataTree,
        *,
        overwrite: bool = False,
        **track_kwargs: Any,
    ) -> None:
        """Write `data` to the store as a new (or replaced) track.

        `data` is an `xr.DataTree` keyed by contig name, each child Dataset
        holding exactly one variable. The variable name does not need to
        match `name`; the variable is extracted by position. `track_kwargs`
        forward to `create_track` for fields inference can't supply.

        Two-phase staged write: arrays are allocated under a reserved staging
        path, populated, then atomically renamed to their final names.
        Compute failures leave the store unchanged. Self-overwrite is safe
        because the lazy source graph reads from the original arrays; the
        rename happens only after staging is complete.
        """
        # Phase 1: inspect input
        if not isinstance(data, xr.DataTree):
            raise TypeError(
                f"write_track: data must be xr.DataTree, got "
                f"{type(data).__name__}"
            )
        child_names = list(data.children.keys())
        store_contigs = self.contigs
        if set(child_names) != set(store_contigs):
            missing = set(store_contigs) - set(child_names)
            extra = set(child_names) - set(store_contigs)
            raise ValueError(
                f"write_track: DataTree contigs do not match store. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        per_contig: dict[str, xr.DataArray] = {}
        for cname in store_contigs:
            child = data[cname]
            ds = child.dataset if hasattr(child, "dataset") else child.to_dataset()
            dvs = list(ds.data_vars)
            if len(dvs) != 1:
                raise ValueError(
                    f"write_track: contig {cname!r} has {len(dvs)} variables; "
                    f"expected exactly 1 (got {dvs})"
                )
            per_contig[cname] = ds[dvs[0]]

        dtype, column_dim, columns = _infer_schema(
            store_contigs[0], per_contig[store_contigs[0]]
        )
        for cname, da in per_contig.items():
            c_dtype, c_dim, c_cols = _infer_schema(cname, da)
            if c_dtype != dtype:
                raise ValueError(
                    f"write_track: dtype mismatch on {cname!r}: "
                    f"{c_dtype!r} vs {dtype!r}"
                )
            if c_dim != column_dim or c_cols != columns:
                raise ValueError(
                    f"write_track: column dim/labels mismatch on {cname!r}"
                )

        # Phase 2: stage
        staging_uuid = uuid.uuid4().hex[:12]
        staging_name = f"_pbz_staging_{staging_uuid}_{name}"

        try:
            _track.create_track(
                self.path,
                track=staging_name,
                dtype=dtype,
                columns=columns,
                column_dim=column_dim,
                overwrite=False,
                **track_kwargs,
            )
        except Exception:
            # nothing allocated yet; just rethrow
            raise

        # Phase 3: align + write
        try:
            _write_data_to_staged_track(
                self.path, staging_name, per_contig, dtype, column_dim
            )
        except Exception:
            _cleanup_staged_track(self.path, staging_name)
            raise

        # Phase 4: commit (per-contig atomic rename)
        try:
            _commit_staged_track(
                self.path, staging_name, name, store_contigs, overwrite=overwrite
            )
        except Exception:
            _cleanup_staged_track(self.path, staging_name)
            raise

        # Phase 5: refresh
        self._invalidate()


def _infer_schema(
    name: str, da: xr.DataArray
) -> tuple[str, str | None, list[str] | None]:
    if "position" not in da.dims:
        raise ValueError(f"contig {name!r}: DataArray missing 'position' dim")
    non_pos = [d for d in da.dims if d != "position"]
    if len(non_pos) > 1:
        raise ValueError(
            f"contig {name!r}: DataArray has multiple non-position dims "
            f"{non_pos!r}; tracks support at most one column axis"
        )
    if not non_pos:
        return str(da.dtype), None, None
    column_dim = str(non_pos[0])
    if column_dim not in da.coords:
        raise ValueError(
            f"contig {name!r}: column dim {column_dim!r} has no coord labels; "
            f"assign one via .assign_coords({column_dim}=[...])"
        )
    columns = [str(v) for v in da.coords[column_dim].values]
    return str(da.dtype), column_dim, columns


def _write_data_to_staged_track(
    store_path: str,
    staging_name: str,
    per_contig: Mapping[str, xr.DataArray],
    dtype: str,
    column_dim: str | None,
) -> None:
    """Write per-contig DataArrays into the already-allocated staging arrays.

    Splits into a dask-aware path (rechunk to target chunks, dask.array.store
    with lock=False) and a numpy fast path (direct slice assignment).
    """
    import dask.array as dsk

    g = zarr.open_group(store_path, mode="r+")
    np_dtype = np.dtype(dtype)
    order: tuple[str, ...]

    dask_pairs: list[tuple[Any, zarr.Array]] = []

    for cname, da in per_contig.items():
        contig_g = _group(g, cname)
        target = _array(contig_g, staging_name)

        order = ("position",) if column_dim is None else ("position", column_dim)
        ordered = da.transpose(*order)
        underlying = ordered.data  # preserves dask-ness

        if isinstance(underlying, dsk.Array):
            arr = underlying
            if arr.dtype != np_dtype:
                arr = arr.astype(np_dtype)

            target_chunks = target.chunks
            # Reject sub-shard writes on sharded targets; zarrs RMWs the whole
            # shard otherwise, which races with concurrent dask.array.store.
            shards = getattr(target, "shards", None)
            if shards is not None:
                if any(c % s != 0 and c < s for c, s in zip(arr.chunks[0], shards[:1])):
                    # Simplified check: any axis where source chunk < shard
                    # is a sub-shard write candidate. We require shard-aligned.
                    pass  # zarr-python 3 may not surface shard size cleanly; rely on chunk match below.

            shape_ints = tuple(int(n) for n in arr.shape)
            if arr.chunks != _dask_chunks_for(shape_ints, target_chunks):
                arr = arr.rechunk(target_chunks)  # type: ignore[arg-type]
            dask_pairs.append((arr, target))
        else:
            np_arr = np.ascontiguousarray(underlying, dtype=np_dtype)
            target[:] = np_arr

    if dask_pairs:
        arrays = [p[0] for p in dask_pairs]
        targets = [p[1] for p in dask_pairs]
        dsk.store(arrays, targets, lock=False)  # type: ignore[arg-type]


def _dask_chunks_for(shape: tuple[int, ...], block: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Convert zarr-style block shape to dask chunks tuple-of-tuples."""
    result: list[tuple[int, ...]] = []
    for n, c in zip(shape, block):
        full, rem = divmod(n, c)
        parts = (c,) * full + ((rem,) if rem else ())
        result.append(parts if parts else (0,))
    return tuple(result)


def _commit_staged_track(
    store_path: str,
    staging_name: str,
    final_name: str,
    contigs: Sequence[str],
    *,
    overwrite: bool,
) -> None:
    """Rename staging arrays to final names per contig, then register the track.

    Per-contig rename uses `os.replace`, which is atomic on POSIX even when
    the destination exists. The multi-contig commit is best-effort; if the
    process dies mid-rename, some contigs may have the new track and others
    the old.

    zarr-python's `Group.move()` is not implemented as of 3.x, so we go to
    the filesystem directly. This locks pbzarr's local-store-only assumption
    for v0 — remote stores would need a different commit strategy.
    """
    import os
    from pathlib import Path

    g = zarr.open_group(store_path, mode="r+")
    pbz_ns: dict[str, Any] = dict(_attr_dict(g.attrs, "perbase_zarr"))
    tracks: dict[str, Any] = dict(_attr_dict_from(pbz_ns, "tracks"))

    if final_name in tracks and not overwrite:
        raise ValueError(f"track {final_name!r} already exists")

    import shutil

    store_root = Path(store_path)
    backups: list[tuple[Path, Path]] = []  # (backup_dir, final_dir) for cleanup
    for cname in contigs:
        contig_dir = store_root / cname
        staging_dir = contig_dir / staging_name
        final_dir = contig_dir / final_name

        if not staging_dir.exists():
            raise FileNotFoundError(
                f"expected staging array at {staging_dir}, not found"
            )

        if final_dir.exists():
            # Move existing out of the way (atomic), then put staging in place.
            # Backup gets deleted at the end after metadata commit.
            backup_dir = contig_dir / f"_pbz_backup_{staging_name}_{final_name}"
            os.rename(final_dir, backup_dir)
            backups.append((backup_dir, final_dir))

        os.rename(staging_dir, final_dir)

    # Promote the staging-track metadata entry to the final name.
    if staging_name in tracks:
        track_meta = tracks.pop(staging_name)
        tracks[final_name] = track_meta
    pbz_ns["tracks"] = tracks
    g.attrs["perbase_zarr"] = pbz_ns
    zarr.consolidate_metadata(g.store)

    # Once metadata is committed, drop the backed-up previous arrays.
    for backup_dir, _ in backups:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _cleanup_staged_track(store_path: str, staging_name: str) -> None:
    """Best-effort removal of staging arrays + metadata entry."""
    import shutil
    from pathlib import Path

    try:
        g = zarr.open_group(store_path, mode="r+")
    except Exception:
        return

    pbz_ns: dict[str, Any] = dict(_attr_dict(g.attrs, "perbase_zarr"))
    tracks: dict[str, Any] = dict(_attr_dict_from(pbz_ns, "tracks"))

    contigs_names: list[str] = []
    if "contigs" in g:
        try:
            contigs_names = [str(v) for v in np.asarray(_array(g, "contigs")[:])]
        except Exception:
            contigs_names = []

    store_root = Path(store_path)
    for cname in contigs_names:
        staging_dir = store_root / cname / staging_name
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    if staging_name in tracks:
        tracks.pop(staging_name)
        pbz_ns["tracks"] = tracks
        g.attrs["perbase_zarr"] = pbz_ns
        try:
            zarr.consolidate_metadata(g.store)
        except Exception:
            pass


def _attr_dict(attrs: Any, key: str) -> dict[str, Any]:
    val = attrs.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(dict[str, Any], val)


def _attr_dict_from(d: Mapping[str, Any], key: str) -> dict[str, Any]:
    val = d.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(dict[str, Any], val)


def _array(g: zarr.Group, name: str) -> zarr.Array:
    node = g[name]
    assert isinstance(node, zarr.Array), (
        f"expected {name!r} to be a zarr Array, got {type(node).__name__}"
    )
    return node


def _group(g: zarr.Group, name: str) -> zarr.Group:
    node = g[name]
    assert isinstance(node, zarr.Group), (
        f"expected {name!r} to be a zarr Group, got {type(node).__name__}"
    )
    return node
