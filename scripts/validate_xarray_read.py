"""Round-trip a format 0.5 store between Rust, released xarray, and pbzarr.

Rust imports the committed d4 cohort; released xarray and ``pbzarr.open``
must agree on it. Python then writes a derived track, and the Rust
``check_dataset`` example must still report the store valid.

Run via ``pixi run validate-roundtrip`` from the repository root.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pbzarr
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
COHORT = [ROOT / "fixtures/d4/cohort_s0.d4", ROOT / "fixtures/d4/cohort_s1.d4"]
# Header order chr2 (500 bp of 100) then chr1 (300 bp of 7).
MULTI_CONTIG = ROOT / "fixtures/d4/multi_contig.d4"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "cohort.pbz"
        pbzarr.create(out, reference={"chr1": 10_000}).close()
        pbzarr.import_d4(
            out, "depth", [(COHORT[0], "s0"), (COHORT[1], "s1")], column_dim="sample"
        )

        released = xr.open_zarr(out, consolidated=False)
        with pbzarr.open(out, chunks=None) as ds:
            opened = ds.to_xarray()
            assert set(released.variables) == set(opened.variables)
            assert dict(released.attrs) == dict(opened.attrs)
            for name in released.variables:
                assert released[name].dims == opened[name].dims, name
                assert released[name].dtype == opened[name].dtype, name
                np.testing.assert_equal(released[name].values, opened[name].values)
            rows = ds.reference.rows("chr1:9990-10000")
            np.testing.assert_equal(
                ds["depth"].isel(position=rows).sel(sample="s1").values,
                np.full(10, 100 + (999 % 50) + 1),
            )
        released.close()

        with pbzarr.open(out, mode="r+") as ds:
            ds.write({"mean_depth": ds["depth"].astype("float32").mean("sample")})

        subprocess.run(
            ["cargo", "build", "--quiet", "-p", "pbzarr", "--example", "check_dataset"],
            cwd=ROOT,
            check=True,
        )
        target = json.loads(
            subprocess.check_output(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=ROOT,
                text=True,
            )
        )["target_directory"]
        report = json.loads(
            subprocess.check_output(
                [str(Path(target) / "debug/examples/check_dataset"), str(out)], text=True
            )
        )
        assert report["status"] == "valid", report["issues"]
        assert report["variables"]["mean_depth"]["logical_support"] == "Supported"

        multi = Path(tmp) / "multi.pbz"
        pbzarr.create(multi, reference={"chr2": 500, "chr1": 300}).close()
        pbzarr.import_d4(multi, "depth", MULTI_CONTIG)
        with pbzarr.open(multi) as ds:
            depth = ds["depth"]
            assert ds.reference.rows("chr1:0-1") == slice(500, 501)
            np.testing.assert_equal(
                depth.isel(position=ds.reference.rows("chr2:499-500")).values, [100]
            )
            np.testing.assert_equal(
                depth.isel(position=ds.reference.rows("chr1:0-1")).values, [7]
            )

    print("round-trip OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
