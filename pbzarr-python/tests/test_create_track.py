"""create_track writes per-contig data arrays + (cohort) coord arrays + updates root tracks map."""
from __future__ import annotations
from pathlib import Path
import zarr
import numpy as np

from pbzarr import create_store, create_track


def test_create_scalar_track(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1", "chr2"], contig_lengths=[1000, 500])

    create_track(str(out), track="mask", dtype="bool")

    g = zarr.open_group(str(out), mode="r")
    pbz_ns = g.attrs["perbase_zarr"]
    assert "mask" in pbz_ns["tracks"]
    meta = pbz_ns["tracks"]["mask"]
    assert meta["dtype"] == "bool"
    assert meta["chunk_size"] == 1_000_000
    assert "column_dim" not in meta

    arr = g["chr1/mask"]
    assert arr.shape == (1000,)
    assert arr.dtype == bool

    arr2 = g["chr2/mask"]
    assert arr2.shape == (500,)


def test_create_cohort_track(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1"], contig_lengths=[2000])

    create_track(
        str(out),
        track="depth",
        dtype="uint16",
        columns=["A", "B", "C"],
        column_dim="sample",
        column_chunk_size=16,
    )

    g = zarr.open_group(str(out), mode="r")
    pbz_ns = g.attrs["perbase_zarr"]
    meta = pbz_ns["tracks"]["depth"]
    assert meta["dtype"] == "uint16"
    assert meta["column_dim"] == "sample"
    assert meta["column_chunk_size"] == 16

    data = g["chr1/depth"]
    assert data.shape == (2000, 3)
    assert data.dtype == np.uint16

    sample = g["chr1/sample"]
    assert list(sample[:]) == ["A", "B", "C"]
