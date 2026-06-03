"""Pure-Python create_store via zarr-python v3.

Matches the on-disk layout that the Rust pbzarr crate writes:
- Root group with `perbase_zarr` attribute namespace.
- 1D string array `contigs` with dimension name `contigs`.
- 1D int64 array `contig_lengths` with dimension name `contigs`.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np
import zarr


PBZ_VERSION = "0.1"


def create_store(
    path: str,
    *,
    contigs: Sequence[str],
    contig_lengths: Sequence[int],
    coordinate_space: str | None = None,
) -> None:
    """Create a new empty PBZ store at `path`.

    The store has no tracks. Add tracks with `pbzarr.create_track(...)`.
    Bulk-import from d4 files via `pbzarr.import_d4(...)`. Read with
    `pbzarr.open(...)` (or `xr.open_datatree(..., engine="zarr")`).
    """
    if len(contigs) != len(contig_lengths):
        raise ValueError(
            f"contigs ({len(contigs)}) and contig_lengths "
            f"({len(contig_lengths)}) must match"
        )

    g = zarr.open_group(path, mode="w-")

    pbz_ns: dict = {"version": PBZ_VERSION, "tracks": {}}
    if coordinate_space is not None:
        pbz_ns["coordinate_space"] = coordinate_space
    g.attrs["perbase_zarr"] = pbz_ns

    names = np.array(list(contigs), dtype=str)
    n = len(names)
    arr = g.create_array(
        "contigs",
        shape=(n,),
        chunks=(n,),
        dtype=str,
        dimension_names=["contigs"],
    )
    arr[:] = names

    lengths = np.array(list(contig_lengths), dtype=np.int64)
    arr = g.create_array(
        "contig_lengths",
        shape=(n,),
        chunks=(n,),
        dtype="int64",
        dimension_names=["contigs"],
    )
    arr[:] = lengths

    # Consolidate so readers can open the store in one fs round-trip instead
    # of walking the group tree per array. Kept in sync by create_track.
    zarr.consolidate_metadata(g.store)
