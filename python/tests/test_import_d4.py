"""Import a small .d4 via PbzStore.import_d4, verify with zarr-python."""
from __future__ import annotations
from pathlib import Path
import zarr

from pbzarr import PbzStore


def test_import_d4_into_scalar_track(tmp_path: Path, write_d4):
    d4 = write_d4("chr1", 1000)
    out = tmp_path / "out.pbz"

    store = PbzStore.create(str(out), contigs=["chr1"], contig_lengths=[1000])
    store.create_track("depth", dtype="int32")
    store.import_d4("depth", sources=[(str(d4), None)])

    g = zarr.open_group(str(out), mode="r")
    arr = g["chr1/depth"]
    assert arr.shape == (1000,)
    assert str(arr.dtype) == "int32"
    for i in range(100):
        v = (i % 50) + 1
        for p in range(i * 10, (i + 1) * 10):
            assert arr[p] == v, f"pos {p}"
