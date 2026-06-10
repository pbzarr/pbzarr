"""PbzStore.write_track: two-phase staging, dask-aware, atomic per-contig."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from pbzarr import PbzStore


def _setup_store_with_depth(tmp_path: Path) -> PbzStore:
    out = tmp_path / "w.pbz"
    store = PbzStore.create(str(out), contigs=["chr1", "chr2"], contig_lengths=[200, 100])
    store.create_track(
        "depth",
        dtype="int32",
        columns=["s1", "s2", "s3", "s4"],
        column_dim="sample",
    )
    g = zarr.open_group(str(out), mode="r+")
    rng = np.random.default_rng(0)
    g["chr1/depth"][:] = rng.integers(0, 50, (200, 4), dtype=np.int32)
    g["chr2/depth"][:] = rng.integers(0, 50, (100, 4), dtype=np.int32)
    return store


def test_write_track_numpy_path(tmp_path: Path):
    """Eager numpy DataTree writes via target[:] = arr fast path."""
    store = _setup_store_with_depth(tmp_path)

    # Build a numpy-backed DataTree of per-contig means.
    chr1_mean = store.tree["chr1"]["depth"].mean("sample").load()
    chr2_mean = store.tree["chr2"]["depth"].mean("sample").load()
    reduced = xr.DataTree.from_dict({
        "chr1": xr.Dataset({"mean_depth": chr1_mean}),
        "chr2": xr.Dataset({"mean_depth": chr2_mean}),
    })

    store.write_track("mean_depth", reduced)

    assert "mean_depth" in store.tracks
    re_opened = PbzStore(store.path)
    np.testing.assert_allclose(
        re_opened.tree["chr1"]["mean_depth"].values, chr1_mean.values
    )
    np.testing.assert_allclose(
        re_opened.tree["chr2"]["mean_depth"].values, chr2_mean.values
    )


def test_write_track_dask_path_with_rechunk(tmp_path: Path):
    """Lazy dask DataTree writes via dask.array.store, rechunked to target."""
    store = _setup_store_with_depth(tmp_path)

    reduced = store.tree.map_over_datasets(
        lambda ds: xr.Dataset({"mean_depth": ds["depth"].mean("sample")})
        if "depth" in ds.data_vars
        else xr.Dataset()
    )
    store.write_track("mean_depth", reduced)

    re_opened = PbzStore(store.path)
    assert "mean_depth" in re_opened.tracks
    expected = store.tree["chr1"]["depth"].mean("sample").values
    np.testing.assert_allclose(
        re_opened.tree["chr1"]["mean_depth"].values, expected
    )


def test_write_track_self_overwrite_safe(tmp_path: Path):
    """Reading from track X and writing back to X with overwrite=True must work."""
    store = _setup_store_with_depth(tmp_path)

    # Add 1 to every value, write back to same track.
    transformed = store.read_track("depth").map_over_datasets(
        lambda ds: xr.Dataset({"depth": ds["depth"] + 1})
        if "depth" in ds.data_vars
        else xr.Dataset()
    )
    original_chr1 = store.tree["chr1"]["depth"].values.copy()

    store.write_track("depth", transformed, overwrite=True)

    re_opened = PbzStore(store.path)
    np.testing.assert_array_equal(
        re_opened.tree["chr1"]["depth"].values, original_chr1 + 1
    )


def test_write_track_rejects_missing_contig(tmp_path: Path):
    store = _setup_store_with_depth(tmp_path)

    chr1_mean = store.tree["chr1"]["depth"].mean("sample").load()
    partial = xr.DataTree.from_dict({"chr1": xr.Dataset({"m": chr1_mean})})
    with pytest.raises(ValueError, match="contigs do not match"):
        store.write_track("m", partial)


def test_write_track_compute_failure_leaves_store_clean(tmp_path: Path):
    """A failing dask compute must not leave staging arrays or partial metadata."""
    store = _setup_store_with_depth(tmp_path)

    import dask.array as dsk

    @dsk.as_gufunc(signature="()->()", output_dtypes=np.int32)
    def boom(x):
        raise RuntimeError("synthetic compute failure")

    # Build a dask-backed DataTree where compute raises.
    def _explode(ds):
        if "depth" not in ds.data_vars:
            return ds
        da = ds["depth"].mean("sample")
        # Wrap the dask array in a map_blocks that raises on compute.
        bad = da.data.map_blocks(
            lambda b: (_ for _ in ()).throw(RuntimeError("synthetic")),
            dtype=np.int32,
        )
        new = xr.DataArray(bad, dims=da.dims, coords=da.coords)
        return xr.Dataset({"bad": new})

    bad_tree = store.tree.map_over_datasets(_explode)
    with pytest.raises(Exception):
        store.write_track("bad", bad_tree)

    # No staging arrays linger, no metadata entry.
    g = zarr.open_group(store.path, mode="r")
    chr1_children = list(g["chr1"].array_keys()) + list(g["chr1"].group_keys())
    assert not any(k.startswith("_pbz_staging_") for k in chr1_children), (
        f"staging arrays leaked: {chr1_children}"
    )
    assert "bad" not in store.tracks
