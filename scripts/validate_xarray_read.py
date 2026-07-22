"""Cross-language round-trip: Rust writes a fixture pbz; xarray reads it.

Run via `pixi run validate-roundtrip` from the repo root.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import xarray as xr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.pbz"
        subprocess.run(
            ["cargo", "run", "--quiet", "--example", "fixture_smoke_store", "--", str(out)],
            check=True,
        )

        dt = xr.open_datatree(out, engine="zarr", consolidated=False)

        # tracks are the children now (not contigs)
        assert set(dt.children) == {"mask", "depth"}, list(dt.children)

        depth_ds = dt["depth"].to_dataset()
        assert depth_ds["values"].dims == ("position", "sample"), depth_ds["values"].dims
        assert list(depth_ds["sample"].values) == ["A", "B", "C"], list(depth_ds["sample"].values)

        mask = dt["mask"].to_dataset()["values"].values   # flat (3000,)
        depth = depth_ds["values"].values                  # flat (3000, 3)

        # chr1 occupies flat positions 0..2000
        for i in range(2_000):
            assert mask[i] == (i % 7 == 0), f"mask[{i}] mismatch (got {mask[i]})"
            assert depth[i, 0] == i, f"depth[{i},0]={depth[i, 0]}"
            assert depth[i, 1] == i * 2, f"depth[{i},1]={depth[i, 1]}"
            assert depth[i, 2] == i * 3, f"depth[{i},2]={depth[i, 2]}"

    print("round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
