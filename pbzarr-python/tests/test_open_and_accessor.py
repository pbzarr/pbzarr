"""pbzarr.open returns an xr.DataTree; .pbz accessor exposes region queries."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

import pbzarr


def _make_store(tmp_path: Path) -> Path:
    out = tmp_path / "x.pbz"
    pbzarr.create_store(
        str(out),
        contigs=["chr1", "chr2"],
        contig_lengths=[2000, 1000],
    )
    pbzarr.create_track(str(out), track="mask", dtype="bool")
    pbzarr.create_track(
        str(out),
        track="depth",
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


def test_accessor_region_with_track_and_sample(tmp_path: Path):
    out = _make_store(tmp_path)
    dt = pbzarr.open(str(out))

    da = dt.pbz.region("chr1:100-200", track="depth", sample="B")
    assert isinstance(da, xr.DataArray)
    assert int(da.sizes["position"]) == 100
    assert "sample" not in da.dims
