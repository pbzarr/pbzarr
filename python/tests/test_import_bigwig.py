"""Import a small .bw via PbzStore.import_bigwig, verify with zarr-python."""
from __future__ import annotations
from pathlib import Path
import zarr

from pbzarr import PbzStore


def test_import_bigwig_into_scalar_track(tmp_path: Path, write_bigwig):
    bw = write_bigwig("chr1", 1000)
    out = tmp_path / "out.pbz"

    store = PbzStore.create(str(out), contigs=["chr1"], contig_lengths=[1000])
    store.create_track("signal", dtype="float32")
    store.import_bigwig("signal", sources=[(str(bw), None)])

    g = zarr.open_group(str(out), mode="r")
    arr = g["chr1/signal"]
    assert arr.shape == (1000,)
    assert str(arr.dtype) == "float32"
    for i in range(100):
        v = float((i % 50) + 1)
        for p in range(i * 10, (i + 1) * 10):
            assert arr[p] == v, f"pos {p}"
