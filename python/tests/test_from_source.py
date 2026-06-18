"""PbzStore.from_d4 / from_bigwig build a store directly from a source file."""
from __future__ import annotations
from pathlib import Path

import pytest
import zarr

from pbzarr import PbzStore


def test_from_d4_single_scalar(tmp_path: Path, write_d4):
    d4 = write_d4("chr1", 1000)
    out = tmp_path / "out.pbz"

    store = PbzStore.from_d4(str(out), str(d4), track="depth")

    assert store.contigs == ["chr1"]
    assert store.contig_length("chr1") == 1000
    meta = store.track_schema("depth")
    assert meta["dtype"] == "int32"
    assert "column_dim" not in meta

    g = zarr.open_group(str(out), mode="r")
    arr = g["chr1/depth"]
    assert arr.shape == (1000,)
    for i in range(100):
        assert arr[i * 10] == (i % 50) + 1


def test_from_d4_cohort_mapping(tmp_path: Path, write_d4):
    a = write_d4("chr1", 1000, sample="A")
    b = write_d4("chr1", 1000, sample="B")
    out = tmp_path / "cohort.pbz"

    store = PbzStore.from_d4(
        str(out), {"A": str(a), "B": str(b)}, track="depth", column_dim="sample"
    )

    meta = store.track_schema("depth")
    assert meta["column_dim"] == "sample"
    assert store.column_labels("depth") == ["A", "B"]

    g = zarr.open_group(str(out), mode="r")
    assert g["chr1/depth"].shape == (1000, 2)


def test_from_d4_contig_mismatch_raises(tmp_path: Path, write_d4):
    a = write_d4("chr1", 1000, sample="A")
    b = write_d4("chr2", 800, sample="B")
    out = tmp_path / "mismatch.pbz"

    with pytest.raises(ValueError, match="share one reference"):
        PbzStore.from_d4(
            str(out), {"A": str(a), "B": str(b)}, track="depth", column_dim="sample"
        )


def test_from_bigwig_single_scalar(tmp_path: Path, write_bigwig):
    bw = write_bigwig("chr1", 1000)
    out = tmp_path / "bw.pbz"

    store = PbzStore.from_bigwig(str(out), str(bw), track="signal")

    assert store.contigs == ["chr1"]
    meta = store.track_schema("signal")
    assert meta["dtype"] == "float32"

    g = zarr.open_group(str(out), mode="r")
    assert g["chr1/signal"].shape == (1000,)
