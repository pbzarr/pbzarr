"""Import a small .d4 via PyO3 into an existing pbz store, verify with zarr-python."""
from __future__ import annotations
from pathlib import Path
import zarr

from pbzarr import create_store, create_track, import_d4


def test_import_d4_into_scalar_track(tmp_path: Path, write_d4):
    d4 = write_d4("chr1", 1000)
    store = tmp_path / "out.pbz"

    create_store(str(store), contigs=["chr1"], contig_lengths=[1000])
    create_track(str(store), track="depth", dtype="uint32")
    import_d4(str(store), track="depth", sources=[(str(d4), None)])

    g = zarr.open_group(str(store), mode="r")
    arr = g["chr1/depth"]
    assert arr.shape == (1000,)
    assert str(arr.dtype) == "uint32"
    for i in range(100):
        v = (i % 50) + 1
        for p in range(i * 10, (i + 1) * 10):
            assert arr[p] == v, f"pos {p}"
