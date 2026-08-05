"""Validate a regular Rust PBZ fixture through released xarray and pbzarr.

Run via ``pixi run validate-roundtrip`` from the repository root.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pbzarr
import xarray as xr


def _dataset(tree: xr.DataTree, track: str) -> xr.Dataset:
    return tree[track].to_dataset(inherit=False)


def _assert_track_equal(
    released: xr.Dataset, opened: xr.Dataset, track: str
) -> None:
    assert set(released.sizes) == set(opened.sizes), track
    assert dict(released.sizes) == dict(opened.sizes), track
    assert set(released.variables) == set(opened.variables), track

    published_attrs = {
        key: value
        for key, value in opened.attrs.items()
        if key != "perbase:kind"
    }
    assert dict(released.attrs) == published_attrs, track
    assert opened.attrs["perbase:kind"] == "track", track

    for name in released.variables:
        left = released[name]
        right = opened[name]
        assert left.dims == right.dims, (track, name)
        assert left.dtype == right.dtype, (track, name)
        np.testing.assert_equal(left.values, right.values, err_msg=(track, name))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.pbz"
        subprocess.run(
            ["cargo", "run", "--quiet", "--example", "fixture_smoke_store", "--", str(out)],
            check=True,
        )

        released = xr.open_datatree(out, engine="zarr", consolidated=False)
        opened = pbzarr.open(out, chunks=None)
        try:
            assert isinstance(opened, xr.DataTree)
            assert dict(released.attrs) == dict(opened.attrs)
            assert set(released.children) == {"depth", "mask"}
            assert set(opened.children) == set(released.children)

            for track in ("mask", "depth"):
                _assert_track_equal(
                    _dataset(released, track), _dataset(opened, track), track
                )

            depth = _dataset(opened, "depth")["values"]
            mask = _dataset(opened, "mask")["values"]
            assert depth.dims == ("position", "sample")
            assert list(_dataset(opened, "depth")["sample"].values) == ["A", "B", "C"]
            assert list(_dataset(opened, "depth")["contigs"].values) == ["chr1", "chr2"]
            np.testing.assert_equal(_dataset(opened, "depth")["offsets"].values, [0, 2000, 3000])

            # The fixture resets its per-contig source pattern at the flat boundary.
            np.testing.assert_equal(depth.isel(position=1999).values, [1999, 3998, 5997])
            np.testing.assert_equal(depth.isel(position=2000).values, [0, 0, 0])
            assert bool(mask.isel(position=1999).values) is False
            assert bool(mask.isel(position=2000).values) is False
        finally:
            opened.close()
            released.close()

    print("round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
