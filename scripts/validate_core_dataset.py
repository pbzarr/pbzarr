"""Cross-language checks for the provisional PBZ root-Dataset reader and imports.

Creates fresh temporary stores. Uses the existing project environment and
respects Cargo's configured target directory. No source stores are modified.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def run_probe(probe, store):
    output = subprocess.run([str(probe), str(store)], check=True, text=True, capture_output=True)
    return json.loads(output.stdout)


def check_rust_imports(writer, probe, destination):
    import zarr
    from dask.callbacks import Callback

    store = destination / "rust-import.pbz"
    subprocess.run([str(writer), str(store)], check=True)
    assert run_probe(probe, store)["status"] == "valid"
    with xr.open_zarr(store, consolidated=False) as ds:
        assert set(ds.data_vars) == {"depth", "zero", "mask"}
        assert {"contig", "contig_length", "sample"} <= set(ds.coords)
        assert "position" not in ds.coords
        assert ds.depth.dims == ("position", "sample")
        np.testing.assert_array_equal(ds.contig.values, ["chr1", "chr2"])
        np.testing.assert_array_equal(ds.contig_length.values, [5, 7])
        np.testing.assert_array_equal(ds.sample.values, ["S1", "S2"])
        np.testing.assert_array_equal(ds.depth.values, np.tile([7, 3], (12, 1)))
        np.testing.assert_array_equal(ds.zero.values, np.zeros(12, dtype=np.uint32))
        np.testing.assert_array_equal(ds["mask"].values, np.zeros(12, dtype=bool))
        assert ds["mask"].dtype == bool
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            counts = (ds.depth >= 5).sum("sample")
        assert not tasks
        np.testing.assert_array_equal(counts.compute().values, np.ones(12, dtype=np.int64))

    metadata = json.loads((store / "depth/zarr.json").read_text())
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [8, 2]
    assert metadata["codecs"][0]["name"] == "sharding_indexed"
    assert metadata["codecs"][0]["configuration"]["chunk_shape"] == [4, 1]
    zero = json.loads((store / "zero/zarr.json").read_text())
    assert zero["codecs"][1]["name"] == "zstd"
    assert zero["codecs"][1]["configuration"]["level"] == 1
    assert zero["attributes"]["notes"] == "Zero remains an ordinary value."
    assert not (store / "zero/c/0").exists()

    # Native consolidation must not hide a successful subsequent Rust import.
    zarr.open_group(store, mode="a").attrs["notes"] = "Preserve this root annotation."
    zarr.consolidate_metadata(store)
    sample_metadata = (store / "sample/zarr.json").read_bytes()
    subprocess.run([str(writer), str(store), "add"], check=True)
    assert (store / "sample/zarr.json").read_bytes() == sample_metadata
    assert "consolidated_metadata" not in json.loads((store / "zarr.json").read_text())
    with xr.open_zarr(store, chunks=None) as ds:
        assert ds.attrs["notes"] == "Preserve this root annotation."
        np.testing.assert_array_equal(ds.added.values, np.tile([12, 20], (12, 1)))
        np.testing.assert_array_equal(ds.depth.values, np.tile([7, 3], (12, 1)))
    assert run_probe(probe, store)["status"] == "valid"
    print("Rust imports passed native Xarray: values, shared coordinates, fills, codecs, shards, and consolidated-store additions.")


def main():
    module_spec = importlib.util.spec_from_file_location(
        "core_examples", ROOT / "docs/proposals/pbz-core-draft/examples.py"
    )
    examples = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(examples)
    subprocess.run(["cargo", "build", "-p", "pbzarr", "--example", "check_core_dataset", "--example", "write_core_dataset"], cwd=ROOT, check=True)
    cargo = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT, text=True
    ))
    probe = Path(cargo["target_directory"]) / "debug/examples/check_core_dataset"
    writer = Path(cargo["target_directory"]) / "debug/examples/write_core_dataset"
    with TemporaryDirectory(prefix="pbz-core-conformance-") as tmp:
        destination = Path(tmp)
        check_rust_imports(writer, probe, destination)
        examples.generate(destination)
        manifest = json.loads((destination / "manifest.json").read_text())
        for relative in manifest["valid"]:
            store = destination / relative
            report = run_probe(probe, store)
            assert report["status"] == "valid", (relative, report)
            with xr.open_zarr(store, consolidated=False) as decoded, xr.open_zarr(
                store, consolidated=False, decode_cf=False
            ) as raw:
                assert set(report["variables"]) == set(raw.variables)
                for name, info in report["variables"].items():
                    assert info["dims"] == list(raw[name].dims), (relative, name)
                    assert info["shape"] == list(raw[name].shape), (relative, name)
                    assert info["role"] == ("coordinate" if name in decoded.coords else "measurement")
                    expected = decoded[name].values if decoded[name].dtype == bool else raw[name].values
                    actual = report["values"][name]
                    if expected.shape:
                        actual = np.asarray(actual, dtype=expected.dtype).reshape(expected.shape)
                    np.testing.assert_array_equal(actual, expected, err_msg=f"{relative}: {name}")
                pointers = raw.attrs["pbz:reference"]
                names = raw[pointers["names"]].values.tolist()
                lengths = raw[pointers["lengths"]].values.tolist()
                assert report["reference"] == [
                    {"name": name, "length": length} for name, length in zip(names, lengths)
                ]
                for dimension, intervals in report["bins"].items():
                    width = int(dimension.removeprefix("bin_"))
                    expected = [
                        {"contig": name, "start": start, "stop": min(start + width, length)}
                        for name, length in zip(names, lengths)
                        for start in range(0, length, width)
                    ]
                    assert intervals == expected, (relative, dimension)
                if "depth_subset" in report:
                    np.testing.assert_array_equal(
                        np.asarray(report["depth_subset"]).reshape(2, 2), raw.depth.values[2:4, 1:3]
                    )
            if store.name == "values-and-missingness.pbz":
                assert report["values"]["large_unsigned"] == 2**53 + 1
                assert report["values"]["large_signed"] == 2**53 + 1
                assert report["values"]["missing_count"][:3] == [0, 1, -1]
        for case in manifest["invalid"]:
            report = run_probe(probe, destination / case["path"])
            assert report["status"] == "invalid", (case, report)
            assert any(i["rule"] in case["rule"].split("/") for i in report["issues"]), (case, report)
        for case in manifest["unsupported"]:
            report = run_probe(probe, destination / case["path"])
            assert report["status"] == "unsupported_version", report
        print(f"Rust conformance passed: {len(manifest['valid'])} valid, "
              f"{len(manifest['invalid'])} invalid, {len(manifest['unsupported'])} unsupported version.")
        print("Compared raw values, decoded Boolean masks, shapes, coordinate roles, bins, and selected depth.")


if __name__ == "__main__":
    main()
