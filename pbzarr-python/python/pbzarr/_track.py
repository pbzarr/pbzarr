"""Pure-Python create_track via zarr-python v3.

Writes the per-contig data array + (for cohort tracks) the column-dim coord
array, then updates the root `perbase_zarr.tracks` map. Matches the layout
that the Rust pbzarr crate writes.
"""
from __future__ import annotations
from typing import Sequence

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
    fill_value=None,
    description: str | None = None,
    source: str | None = None,
) -> None:
    """Register a new track in the store at `path`.

    Pass `columns=[...]` for a 2D cohort track; omit for a 1D scalar track.
    `column_dim` defaults to `"column"` when `columns` is given.
    """
    g = zarr.open_group(path, mode="r+")

    pbz_ns = dict(g.attrs.get("perbase_zarr", {}))
    tracks = dict(pbz_ns.get("tracks", {}))
    if track in tracks:
        raise ValueError(f"track {track!r} already exists")

    contigs = list(g["contigs"][:])
    contig_lengths = list(map(int, g["contig_lengths"][:]))

    is_cohort = columns is not None
    chunk = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    col_chunk = column_chunk_size if column_chunk_size is not None else (
        DEFAULT_COLUMN_CHUNK_SIZE if is_cohort else None
    )
    dim_name = column_dim if column_dim is not None else (
        "column" if is_cohort else None
    )

    np_dtype = np.dtype(dtype)
    codecs = default_data_codecs()

    for name, length in zip(contigs, contig_lengths):
        contig_g = g.require_group(name)

        if is_cohort:
            n_cols = len(columns)
            data_shape = (length, n_cols)
            data_chunks = (
                max(1, min(chunk, length)),
                max(1, min(col_chunk, n_cols)),
            )
            dims = ["position", dim_name]
        else:
            data_shape = (length,)
            data_chunks = (max(1, min(chunk, length)),)
            dims = ["position"]

        contig_g.create_array(
            track,
            shape=data_shape,
            chunks=data_chunks,
            dtype=np_dtype,
            dimension_names=dims,
            compressors=codecs,
        )

        if is_cohort and dim_name not in contig_g:
            labels = np.array(list(columns), dtype=str)
            coord = contig_g.create_array(
                dim_name,
                shape=(n_cols,),
                chunks=(n_cols,),
                dtype=str,
                dimension_names=[dim_name],
            )
            coord[:] = labels

    # root tracks map update
    meta: dict = {"dtype": dtype, "chunk_size": chunk}
    if is_cohort:
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
