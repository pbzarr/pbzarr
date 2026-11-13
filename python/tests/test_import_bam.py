from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

import pbzarr

READS_BAM = Path(__file__).resolve().parents[2] / "fixtures" / "bam" / "reads.bam"


def test_import_bam_depth_single_source_matches_hand_computed_vector(tmp_path):
    # Spans that count under the default filter, from fixtures/bam/README.md.
    spans = [
        (10, 30), (30, 45), (40, 50), (55, 65), (70, 75), (85, 90),
        (10, 30), (50, 55), (55, 60), (12, 22), (90, 98),
    ]
    expected = np.zeros(160, dtype=np.int32)
    for start, end in spans:
        expected[start:end] += 1

    destination = tmp_path / "depth.pbz"
    pbzarr.create(destination, reference={"ref1": 100, "ref2": 60}).close()

    report = pbzarr.import_bam(destination, "depth", READS_BAM, mode="depth")
    assert set(report) == {
        "positions",
        "chunks_written",
        "chunks_skipped",
        "bytes_written",
        "wall_seconds",
    }
    assert report["positions"] == 160
    assert report["chunks_written"] == 1
    assert report["chunks_skipped"] == 0
    assert report["bytes_written"] > 0

    with pbzarr.open(destination, chunks=None) as dataset:
        assert isinstance(dataset, xr.Dataset)
        assert dataset["depth"].dims == ("position",)
        assert dataset["depth"].dtype == np.dtype("int32")
        assert dataset["depth"].values.tolist() == expected.tolist()
