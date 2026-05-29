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


def test_sharded_track_round_trips(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1"], contig_lengths=[10_000])
    create_track(
        str(out),
        track="depth",
        dtype="uint32",
        chunk_size=1_000,
        shard_size=4_000,
    )

    g = zarr.open_group(str(out), mode="r+")
    arr = g["chr1/depth"]
    arr[:] = np.arange(10_000, dtype=np.uint32)

    g2 = zarr.open_group(str(out), mode="r")
    assert (g2["chr1/depth"][:] == np.arange(10_000)).all()


import json


def _array_codecs(store: str, contig: str, track: str) -> list[dict]:
    """Read the codec list from a track array's Zarr v3 metadata."""
    meta = json.loads((Path(store) / contig / track / "zarr.json").read_text())
    return meta["codecs"]


def _blosc_config(codecs: list[dict]) -> dict | None:
    for c in codecs:
        if c["name"] == "blosc":
            return c["configuration"]
    return None


def test_default_codecs_are_blosc_zstd5(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1"], contig_lengths=[1000])
    create_track(str(out), track="depth", dtype="uint16")

    cfg = _blosc_config(_array_codecs(str(out), "chr1", "depth"))
    assert cfg is not None, "default should apply a Blosc codec"
    assert cfg["cname"] == "zstd"
    assert cfg["clevel"] == 5


def test_compressors_override_is_applied(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1"], contig_lengths=[1000])

    from zarr.codecs import BloscCodec, BloscShuffle
    create_track(
        str(out),
        track="depth",
        dtype="uint16",
        compressors=[BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle)],
    )

    cfg = _blosc_config(_array_codecs(str(out), "chr1", "depth"))
    assert cfg is not None
    assert cfg["clevel"] == 1, "override clevel should win over the default 5"


def test_empty_compressors_means_uncompressed(tmp_path: Path):
    out = tmp_path / "t.pbz"
    create_store(str(out), contigs=["chr1"], contig_lengths=[1000])
    create_track(str(out), track="depth", dtype="uint16", compressors=[])

    codecs = _array_codecs(str(out), "chr1", "depth")
    assert _blosc_config(codecs) is None, "empty list must disable compression"
    assert [c["name"] for c in codecs] == ["bytes"]
