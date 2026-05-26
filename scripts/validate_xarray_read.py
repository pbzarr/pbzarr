"""Cross-language round-trip: Rust writes a fixture pbz; xarray reads it.

Run via `pixi run validate-roundtrip` from the repo root.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import xarray as xr


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke.pbz"
        subprocess.run(
            ["cargo", "run", "--quiet", "--example", "fixture_smoke_store", "--", str(out)],
            check=True,
        )

        dt = xr.open_datatree(out, engine="zarr")

        # root has contigs + contig_lengths
        assert "contigs" in dt, list(dt.data_vars) + list(dt.coords)
        names = list(dt["contigs"].values)
        assert names == ["chr1", "chr2"], names

        chr1 = dt["chr1"].to_dataset()
        assert "mask" in chr1, list(chr1.data_vars)
        assert chr1["mask"].dims == ("position",), chr1["mask"].dims
        assert "depth" in chr1
        assert chr1["depth"].dims == ("position", "sample"), chr1["depth"].dims
        assert list(chr1.coords["sample"].values) == ["A", "B", "C"], \
            list(chr1.coords["sample"].values)

        # values
        mask = chr1["mask"].values
        for i in range(2_000):
            assert mask[i] == (i % 7 == 0), f"mask[{i}] mismatch (got {mask[i]})"
        depth = chr1["depth"].values
        for i in range(2_000):
            assert depth[i, 0] == i, f"depth[{i},0]={depth[i,0]}"
            assert depth[i, 1] == i * 2, f"depth[{i},1]={depth[i,1]}"
            assert depth[i, 2] == i * 3, f"depth[{i},2]={depth[i,2]}"

    print("round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
