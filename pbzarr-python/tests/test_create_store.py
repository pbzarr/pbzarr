"""create_store writes contigs + contig_lengths + root perbase_zarr attrs."""
from __future__ import annotations
from pathlib import Path
import zarr

from pbzarr import create_store


def test_create_store_writes_contigs_and_root_attrs(tmp_path: Path):
    out = tmp_path / "t.pbz"

    create_store(
        str(out),
        contigs=["chr1", "chr2"],
        contig_lengths=[1_000_000, 500_000],
        coordinate_space="GRCh38",
    )

    g = zarr.open_group(str(out), mode="r")

    assert list(g["contigs"][:]) == ["chr1", "chr2"]
    assert list(g["contig_lengths"][:]) == [1_000_000, 500_000]

    pbz_ns = g.attrs["perbase_zarr"]
    assert pbz_ns["version"] == "0.1"
    assert pbz_ns["coordinate_space"] == "GRCh38"
    assert pbz_ns["tracks"] == {}
