"""Pure-Python create_track via zarr-python v3.

Writes the per-contig data array + (for cohort tracks) the column-dim coord
array, then updates the root `perbase_zarr.tracks` map. Matches the layout
that the Rust pbzarr crate writes.
"""
from __future__ import annotations
from typing import Any, Sequence, cast

import numpy as np
import zarr

from ._compression import default_data_codecs

DEFAULT_CHUNK_SIZE = 1_000_000
DEFAULT_COLUMN_CHUNK_SIZE = 16


def create_track(
    path: str,
    *,
    track: str,
    dtype: str,
    columns: Sequence[str] | None = None,
    column_dim: str | None = None,
    chunk_size: int | None = None,
    column_chunk_size: int | None = None,
    shard_size: int | None = None,
    shard_column_size: int | None = None,
    compressors: Sequence | None = None,
    fill_value=None,
    description: str | None = None,
    source: str | None = None,
    overwrite: bool = False,
) -> None:
    """Register a new track in the store at `path`.

    Pass `columns=[...]` for a 2D cohort track; omit for a 1D scalar track.
    `column_dim` defaults to `"column"` when `columns` is given.
    `compressors` overrides the data-array codecs; `None` uses the library
    default (Blosc zstd-5, byte shuffle) and `[]` writes uncompressed.
    The chosen codec is recorded on the array and reused by any later
    writer that fills it. If the track will be populated through
    `import_d4` (the Rust/zarrs path), pick a codec `zarrs` can also
    encode, since the importer encodes through the pipeline recorded here.
    Pass `overwrite=True` to replace an existing track of the same name;
    the existing per-contig data arrays are deleted before recreation.
    """
    g = zarr.open_group(path, mode="r+")

    pbz_ns: dict[str, Any] = dict(_attr_dict(g.attrs, "perbase_zarr"))
    tracks: dict[str, Any] = dict(_attr_dict_from(pbz_ns, "tracks"))

    contigs = [str(v) for v in np.asarray(_array(g, "contigs")[:])]
    contig_lengths = [int(v) for v in np.asarray(_array(g, "contig_lengths")[:])]

    if track in tracks:
        if not overwrite:
            raise ValueError(f"track {track!r} already exists")
        for name in contigs:
            cg = _group(g, name)
            if track in cg:
                del cg[track]
        tracks.pop(track)

    chunk = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    np_dtype = np.dtype(dtype)
    codecs = list(compressors) if compressors is not None else default_data_codecs()

    if columns is not None:
        n_cols = len(columns)
        dim_name = column_dim if column_dim is not None else "column"
        col_chunk = (
            column_chunk_size if column_chunk_size is not None
            else DEFAULT_COLUMN_CHUNK_SIZE
        )
    else:
        n_cols = 0
        dim_name = None
        col_chunk = None

    for name, length in zip(contigs, contig_lengths):
        contig_g = g.require_group(name)
        inner_pos = max(1, min(chunk, length))

        if columns is not None:
            assert dim_name is not None and col_chunk is not None
            inner_col = max(1, min(col_chunk, n_cols))
            data_shape: tuple[int, ...] = (length, n_cols)
            data_chunks: tuple[int, ...] = (inner_pos, inner_col)
            dims = ["position", dim_name]
            if shard_size is not None:
                col_shard = shard_column_size or n_cols
                # Shard must be an exact multiple of inner chunk size; do not
                # clamp to contig length (zarr handles partial edge shards).
                pos_shard = max(inner_pos, (shard_size // inner_pos) * inner_pos)
                data_shards: tuple[int, ...] | None = (
                    pos_shard,
                    max(1, min(col_shard, n_cols)),
                )
            else:
                data_shards = None
        else:
            data_shape = (length,)
            data_chunks = (inner_pos,)
            dims = ["position"]
            if shard_size is not None:
                pos_shard = max(inner_pos, (shard_size // inner_pos) * inner_pos)
                data_shards = (pos_shard,)
            else:
                data_shards = None

        contig_g.create_array(
            track,
            shape=data_shape,
            chunks=data_chunks,
            shards=data_shards,
            dtype=np_dtype,
            dimension_names=dims,
            compressors=codecs,
        )

        if columns is not None:
            assert dim_name is not None
            labels = np.array(list(columns), dtype=str)
            if dim_name in contig_g:
                coord_existing = _array(contig_g, dim_name)
                existing = [str(v) for v in np.asarray(coord_existing[:])]
                if existing != list(labels):
                    raise ValueError(
                        f"column dim {dim_name!r} already exists on {name!r} "
                        f"with different labels; pick a different column_dim"
                    )
            else:
                coord = contig_g.create_array(
                    dim_name,
                    shape=(n_cols,),
                    chunks=(n_cols,),
                    dtype=str,
                    dimension_names=[dim_name],
                )
                coord[:] = labels

    meta: dict[str, Any] = {"dtype": dtype, "chunk_size": chunk}
    if columns is not None:
        meta["column_dim"] = dim_name
        if col_chunk is not None:
            meta["column_chunk_size"] = col_chunk
    if shard_size is not None:
        meta["shard_size"] = shard_size
    if shard_column_size is not None:
        meta["shard_column_size"] = shard_column_size
    if fill_value is not None:
        meta["fill_value"] = fill_value
    if description is not None:
        meta["description"] = description
    if source is not None:
        meta["source"] = source

    tracks[track] = meta
    pbz_ns["tracks"] = tracks
    g.attrs["perbase_zarr"] = pbz_ns

    # Refresh consolidated metadata so readers benefit from the fast path.
    zarr.consolidate_metadata(g.store)


def _attr_dict(attrs: Any, key: str) -> dict[str, Any]:
    val = attrs.get(key, {})
    if not isinstance(val, dict):
        return {}
    return cast(dict[str, Any], val)


def _attr_dict_from(d: dict[str, Any], key: str) -> dict[str, Any]:
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
