"""Python writes; Rust opens + reads. Validates layout consistency."""
from __future__ import annotations
from pathlib import Path
import subprocess
import zarr

from pbzarr import PbzStore


def test_python_writes_rust_reads(tmp_path: Path):
    out = tmp_path / "py.pbz"

    store = PbzStore.create(
        str(out),
        contigs=["chr1", "chr2"],
        contig_lengths=[100, 50],
        coordinate_space="GRCh38",
    )
    store.create_track("mask", dtype="bool")
    store.create_track(
        "depth",
        dtype="uint16",
        columns=["A", "B", "C"],
        column_dim="sample",
    )

    # write depth data so the Rust side can assert on it
    g = zarr.open_group(str(out), mode="r+")
    depth = g["chr1/depth"]
    for i in range(100):
        depth[i, 0] = i
        depth[i, 1] = i * 2
        depth[i, 2] = i * 3

    result = subprocess.run(
        [
            "cargo", "run", "--quiet",
            "--example", "validate_py_written_store",
            "--", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Python→Rust round-trip OK" in result.stdout, result.stdout
