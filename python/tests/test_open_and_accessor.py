"""pbzarr.open returns an xr.DataTree; .pbz accessor exposes region queries."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

import pbzarr


def _make_store(tmp_path: Path) -> Path:
    out = tmp_path / "x.pbz"
    store = pbzarr.PbzStore.create(
        str(out),
        contigs=["chr1", "chr2"],
        contig_lengths=[2000, 1000],
    )
    store.create_track("mask", dtype="bool")
    store.create_track(
        "depth",
        dtype="uint16",
        columns=["A", "B", "C"],
        column_dim="sample",
    )

    g = zarr.open_group(str(out), mode="r+")
    g["chr1/depth"][:1000, :] = np.tile(np.arange(1000, dtype=np.uint16)[:, None], (1, 3))
    g["chr1/mask"][:1000] = np.arange(1000) % 7 == 0
    return out


def test_open_returns_datatree_with_tracks(tmp_path: Path):
    out = _make_store(tmp_path)

    dt = pbzarr.open(str(out))
    assert isinstance(dt, xr.DataTree)
    assert sorted(dt.pbz.tracks) == ["depth", "mask"]


def test_accessor_region_returns_dataset(tmp_path: Path):
    out = _make_store(tmp_path)
    dt = pbzarr.open(str(out))

    ds = dt.pbz.region("chr1:100-200")
    assert isinstance(ds, xr.Dataset)
    assert int(ds.sizes["position"]) == 100
    assert "depth" in ds.data_vars
    assert "mask" in ds.data_vars
    assert ds["depth"].dims == ("position", "sample")


def test_accessor_region_with_track_and_column(tmp_path: Path):
    out = _make_store(tmp_path)
    dt = pbzarr.open(str(out))

    da = dt.pbz.region("chr1:100-200", track="depth", column="B")
    assert isinstance(da, xr.DataArray)
    assert int(da.sizes["position"]) == 100
    assert "sample" not in da.dims


def test_store_region_delegates_to_accessor(tmp_path: Path):
    out = _make_store(tmp_path)
    store = pbzarr.PbzStore(str(out))

    da = store.region("chr1:100-200", track="depth", column="B")
    assert isinstance(da, xr.DataArray)
    assert int(da.sizes["position"]) == 100


def test_assign_column_labels(tmp_path: Path):
    out = _make_store(tmp_path)
    store = pbzarr.PbzStore(str(out))

    labeled = store.read_track("depth").pbz.assign_column_labels(
        "sample",
        pop={"A": "POP_A", "B": "POP_A", "C": "POP_B"},
    )
    chr1 = labeled["chr1"].to_dataset()
    assert "pop" in chr1.coords
    assert list(chr1["pop"].values) == ["POP_A", "POP_A", "POP_B"]


def test_assign_column_labels_missing_key_raises(tmp_path: Path):
    out = _make_store(tmp_path)
    store = pbzarr.PbzStore(str(out))

    import pytest
    with pytest.raises(ValueError, match="missing keys"):
        store.read_track("depth").pbz.assign_column_labels(
            "sample",
            pop={"A": "POP_A"},  # missing B and C
        )
