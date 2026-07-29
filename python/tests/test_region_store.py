"""Cross-language region-store build: Rust `to_pbz` must match the Python serial
`build_peak_store`, and the resulting store must reduce correctly via `PeakStore`.
"""
from __future__ import annotations

import numpy as np
import zarr

import pbzarr  # noqa: F401 - registers the .pbz accessor
from pbzarr._peakstore import PeakStore, build_peak_store

from _helpers import rust_fixture_store

# Deliberately unsorted, disjoint, spanning both contigs and a scalar + 2D track.
INTERVALS = [
    ("chr1", 100, 250),
    ("chr2", 5, 55),
    ("chr1", 1000, 1003),
    ("chr1", 300, 400),
]


def test_rust_to_pbz_matches_python_builder(tmp_path):
    src = rust_fixture_store(tmp_path)
    ds = pbzarr.PbzStore(str(src)).dataset()

    py_ps = build_peak_store(ds, INTERVALS, str(tmp_path / "py.pbz"))
    rust_ps = ds.pbz.region_view(INTERVALS).to_pbz(str(tmp_path / "rust.pbz"))

    # Provenance identical.
    np.testing.assert_array_equal(py_ps.offsets, rust_ps.offsets)
    assert list(py_ps.region_contig) == list(rust_ps.region_contig)
    np.testing.assert_array_equal(py_ps.region_start, rust_ps.region_start)
    np.testing.assert_array_equal(py_ps.region_stop, rust_ps.region_stop)
    np.testing.assert_array_equal(py_ps.region_input_index, rust_ps.region_input_index)
    assert py_ps.features() == rust_ps.features()

    # `values` byte-identical per track.
    py_root = zarr.open_group(str(tmp_path / "py.pbz"), mode="r")
    rust_root = zarr.open_group(str(tmp_path / "rust.pbz"), mode="r")
    for name in py_ps.features():
        pv = np.asarray(py_root[name]["values"][:])
        rv = np.asarray(rust_root[name]["values"][:])
        np.testing.assert_array_equal(pv, rv, err_msg=f"values differ: {name}")

    # Reductions match through the PeakStore reader.
    for red in ("mean", "sum", "min", "max"):
        pa = getattr(py_ps, red)().compute()
        ra = getattr(rust_ps, red)().compute()
        for name in py_ps.features():
            np.testing.assert_allclose(
                np.asarray(pa[name]), np.asarray(ra[name]), equal_nan=True,
                err_msg=f"{red} differs: {name}",
            )


def test_to_pbz_requires_source_path(tmp_path):
    """A region view over a bare (non-store) dataset can't drive the Rust builder."""
    import xarray as xr

    ds = xr.Dataset(
        {"x": ("position", np.arange(10.0))},
        coords={"contigs": ("contig", np.array(["chr1"])),
                "offsets": ("contig_boundary", np.array([0, 10]))},
    )
    rv = ds.pbz.region_view([("chr1", 1, 5)])
    try:
        rv.to_pbz(str(tmp_path / "out.pbz"))
    except ValueError as e:
        assert "on-disk source" in str(e)
    else:
        raise AssertionError("expected ValueError for missing source path")
