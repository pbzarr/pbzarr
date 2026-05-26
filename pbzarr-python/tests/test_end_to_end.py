"""Full Python user journey: create_store -> create_track -> import_d4 -> open + slice."""
from __future__ import annotations
from pathlib import Path
import numpy as np

import pbzarr


def test_end_to_end_import_then_read(tmp_path: Path, write_d4):
    d4 = write_d4("chr1", 1000)
    out = tmp_path / "e2e.pbz"

    pbzarr.create_store(
        str(out),
        contigs=["chr1"],
        contig_lengths=[1000],
        coordinate_space="GRCh38",
    )
    pbzarr.create_track(str(out), track="depth", dtype="uint32")
    pbzarr.import_d4(
        str(out),
        track="depth",
        sources=[(str(d4), None)],
    )

    dt = pbzarr.open(str(out))
    da = dt.pbz.region("chr1:0-500", track="depth")
    arr = da.values
    assert arr.shape == (500,)
    for i in range(50):
        v = (i % 50) + 1
        for p in range(i * 10, (i + 1) * 10):
            assert arr[p] == v, f"pos {p}"


def test_end_to_end_cohort_via_zarr_python_writes(tmp_path: Path):
    """Cohort write path that doesn't go through import_d4: user opens the
    zarr-python group and writes directly. Documents the v0 pattern from
    the design doc."""
    import zarr

    out = tmp_path / "cohort.pbz"
    pbzarr.create_store(str(out), contigs=["chr1"], contig_lengths=[100])
    pbzarr.create_track(
        str(out),
        track="meth",
        dtype="float32",
        columns=["A", "B"],
        column_dim="sample",
    )

    g = zarr.open_group(str(out), mode="r+")
    g["chr1/meth"][:, 0] = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    g["chr1/meth"][:, 1] = np.linspace(1.0, 0.0, 100, dtype=np.float32)

    dt = pbzarr.open(str(out))
    da = dt.pbz.region("chr1:0-10", track="meth")
    assert da.shape == (10, 2)
    assert da.sel(sample="A")[0].item() == 0.0
    assert da.sel(sample="B")[0].item() == 1.0
