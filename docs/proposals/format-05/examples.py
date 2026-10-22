"""Native Zarr examples for the PBZ core draft; not a production PBZ validator.

Run from the repository root with:
    pixi run python docs/proposals/format-05/examples.py

By default stores are temporary. --output NEW_DIRECTORY keeps them for review.
Run pixi run validate-dataset to check the generated corpus with Rust.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import xarray as xr
import zarr


def root_attrs():
    return {
        "zarr_conventions": [{
            "uuid": "d5bd7bea-0388-4e0a-9844-707c034e9add", "name": "pbz",
        }],
        "pbz:version": "0.5",
        "pbz:reference": {"names": "contig", "lengths": "contig_length"},
    }


def cohort():
    raw = (np.arange(36).reshape(12, 3) % 13).astype("uint16")
    ds = xr.Dataset(
        {"depth": (("position", "sample"), da.from_array(raw, chunks=(4, 3)))},
        coords={
            "contig": np.asarray(["chr1", "chr2"], dtype=object),
            "contig_length": ("contig", np.asarray([5, 7], dtype="int64")),
            "sample": np.asarray(["sample_1", "sample_2", "sample_3"], dtype=object),
            "threshold": np.asarray([5, 10], dtype="int64"),
        },
        attrs=root_attrs(),
    )
    ds["callable"] = (ds.depth >= ds.threshold).transpose("position", "sample", "threshold")
    ds["callable_count"] = ds.callable.sum("sample", dtype="uint64")
    ds["callable_count"].attrs["notes"] = "Number of samples meeting each depth threshold."
    ds["coverage"] = (("bin_4", "sample"), np.arange(12, dtype="float32").reshape(4, 3))
    ds["signal"] = (("bin_6", "sample"), np.arange(9, dtype="float32").reshape(3, 3))
    ds["mean_depth"] = ds.depth.mean("position")
    ds["total_callable_count"] = ds.callable_count.sum(dtype="uint64")
    return ds, raw


def save(ds, path, *, encoding=None, expected=None):
    """Write through Xarray and compare the decoded result with explicit intent."""
    options = {name: {"_FillValue": None} for name in ds.variables}
    for name, updates in (encoding or {}).items():
        options[name].update(updates)
    ds.to_zarr(path, zarr_format=3, consolidated=False, mode="w-", encoding=options)
    if expected is not None:
        with xr.open_zarr(path, consolidated=False) as reopened:
            xr.testing.assert_identical(reopened.load(), expected.compute())


def edit_metadata(path, array, change):
    file = (path / array / "zarr.json") if array else (path / "zarr.json")
    metadata = json.loads(file.read_text())
    change(metadata)
    file.write_text(json.dumps(metadata, indent=2) + "\n")


def generate(destination):
    valid_dir = destination / "valid"
    invalid_dir = destination / "invalid"
    valid_dir.mkdir()
    invalid_dir.mkdir()
    manifest = {"format": "0.5", "valid": [], "invalid": [], "unsupported": []}

    tasks = []
    with Callback(pretask=lambda key, *args: tasks.append(key)):
        base, raw = cohort()
    assert not tasks, "Constructing the small expression must not compute measurements"

    def valid(name, ds, *, encoding=None, expected=None):
        path = valid_dir / f"{name}.pbz"
        save(ds, path, encoding=encoding, expected=ds if expected is None else expected)
        reference = ds.attrs["pbz:reference"]
        names_metadata = json.loads((path / reference["names"] / "zarr.json").read_text())
        lengths_metadata = json.loads((path / reference["lengths"] / "zarr.json").read_text())
        assert names_metadata["data_type"] == "string"
        assert lengths_metadata["data_type"] == "int64"
        manifest["valid"].append(str(path.relative_to(destination)))
        return path

    canonical = valid("cohort", base)
    with xr.open_zarr(canonical, consolidated=False) as reopened:
        np.testing.assert_array_equal(
            reopened.callable_count.compute(), (raw[:, :, None] >= [5, 10]).sum(axis=1)
        )
        assert set(reopened.coords) == {"contig", "contig_length", "sample", "threshold"}
        assert reopened.callable.dtype == np.dtype("bool")
        assert reopened.callable_count.dtype == np.dtype("uint64")
    scalar_metadata = json.loads((canonical / "total_callable_count/zarr.json").read_text())
    assert scalar_metadata["shape"] == []
    assert scalar_metadata.get("dimension_names") in (None, [])

    renamed = base.rename({"contig": "sequence", "contig_length": "sequence_length"})
    renamed.attrs = copy.deepcopy(base.attrs)
    renamed.attrs["pbz:reference"] = {"names": "sequence", "lengths": "sequence_length"}
    valid("renamed-reference", renamed)

    non_genomic = base.drop_vars(["depth", "callable", "callable_count", "coverage", "signal"])
    valid("non-genomic", non_genomic)

    for name, names, lengths in [
        ("zero-length-contigs", ["empty_before", "chr1", "empty_after"], [0, 5, 0]),
        ("empty-reference", [], []),
    ]:
        positions = sum(lengths)
        bins = sum((length + 3) // 4 for length in lengths)
        ds = xr.Dataset(
            {
                "depth": (("position", "sample"), np.zeros((positions, 1), dtype="uint16")),
                "coverage": (("bin_4", "sample"), np.zeros((bins, 1), dtype="float32")),
            },
            coords={
                # An empty object array would otherwise serialize as float64.
                "contig": np.asarray(names, dtype=np.dtypes.StringDType()),
                "contig_length": ("contig", np.asarray(lengths, dtype="int64")),
                "sample": np.asarray(["sample_1"], dtype=object),
            },
            attrs=root_attrs(),
        )
        # This Xarray version reopens variable-length strings as object arrays.
        decoded_strings = ds.assign_coords(contig=ds.contig.astype(object))
        valid(name, ds, expected=decoded_strings)

    regional = np.zeros(12, dtype="int64")
    regional[2:4] = (raw[2:4] >= 5).sum(axis=1)
    regional_ds = xr.Dataset(
        {"regional_count": ("position", regional)},
        coords={"contig": base.contig, "contig_length": base.contig_length},
        attrs=root_attrs(),
    )
    valid("regional-fill", regional_ds, encoding={"regional_count": {"fill_value": 0}})

    missing = xr.Dataset(
        {
            "ordinary_zero": ("position", np.zeros(12, dtype="int16")),
            "missing_count": ("position", np.tile(np.array([0, 1, -1], dtype="int16"), 4)),
            "large_unsigned": xr.DataArray(np.uint64(2**53 + 1)),
            "large_signed": xr.DataArray(np.int64(2**53 + 1)),
        },
        coords={"contig": base.contig, "contig_length": base.contig_length},
        attrs=root_attrs(),
    )
    decoded = missing.copy(deep=True)
    decoded["missing_count"] = ("position", np.tile(np.array([0, 1, np.nan], dtype="float32"), 4))
    missing_path = valid("values-and-missingness", missing, expected=decoded, encoding={
        "ordinary_zero": {"fill_value": 0},
        "missing_count": {"fill_value": 0, "_FillValue": -1},
    })
    with xr.open_zarr(missing_path, consolidated=False) as reopened:
        for name, dtype in [("large_unsigned", "uint64"), ("large_signed", "int64")]:
            assert reopened[name].dtype == np.dtype(dtype)
            assert reopened[name].compute().item() == 2**53 + 1

    def invalid_copy(name, rule, mutate):
        path = invalid_dir / f"{name}.pbz"
        shutil.copytree(canonical, path)
        mutate(path)
        manifest["invalid"].append({"path": str(path.relative_to(destination)), "rule": rule})

    def invalid_native(name, rule, ds):
        path = invalid_dir / f"{name}.pbz"
        save(ds, path)
        manifest["invalid"].append({"path": str(path.relative_to(destination)), "rule": rule})

    metadata_mutations = [
        ("missing-reference", "R2", "", lambda m: m["attributes"].pop("pbz:reference")),
        ("same-reference-pointer", "R2", "", lambda m: m["attributes"]["pbz:reference"].update(lengths="contig")),
        ("dangling-reference-pointer", "R2", "", lambda m: m["attributes"]["pbz:reference"].update(names="absent")),
        ("lengths-as-track", "R3", "", lambda m: m["attributes"].pop("coordinates")),
        ("short-position", "G2/D2", "depth", lambda m: m.update(shape=[11, 3])),
        ("unnamed-axis", "D1", "depth", lambda m: m.update(dimension_names=["position", None])),
        ("global-binning", "G2", "coverage", lambda m: m.update(shape=[3, 3])),
    ]
    for name, rule, array, change in metadata_mutations:
        invalid_copy(name, rule, lambda path: edit_metadata(path, array, change))

    for name, rule, array, values in [
        ("duplicate-contigs", "R4", "contig", ["chr1", "chr1"]),
        ("negative-length", "R6", "contig_length", [5, -7]),
        ("duplicate-samples", "D3", "sample", ["sample_1", "sample_1", "sample_3"]),
    ]:
        invalid_copy(name, rule, lambda path: zarr.open_group(path, mode="a")[array].__setitem__(slice(None), values))

    invalid_native("floating-lengths", "R5", base.assign_coords(contig_length=base.contig_length.astype("float64")))
    transposed = base.copy()
    transposed["depth"] = transposed.depth.transpose("sample", "position")
    invalid_native("genomic-axis-second", "G3", transposed)
    two_axes = base.assign(two_axes=(("position", "bin_4"), np.zeros((12, 4), dtype="uint8")))
    invalid_native("two-genomic-axes", "G3", two_axes)
    invalid_native("stored-position-coordinate", "G4", base.assign_coords(position=np.arange(12)))
    invalid_native("noncanonical-bin-name", "G1", base.rename({"bin_4": "bin_04"}))
    invalid_native("unlabeled-sample-axis", "D3", base.drop_vars("sample"))

    unknown = destination / "unsupported-version.pbz"
    shutil.copytree(canonical, unknown)
    edit_metadata(unknown, "", lambda m: m["attributes"].update({"pbz:version": "unknown-example-version"}))
    manifest["unsupported"].append({"path": unknown.name, "reason": "unknown format version"})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(manifest['valid'])} native Xarray round trips; no Dask tasks during expression construction.")
    print(f"Generated {len(manifest['invalid'])} invalid fixtures and one unsupported-version fixture.")
    print("This generator checks native round trips; the separate Rust harness checks PBZ rejection expectations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory in which to retain the example stores")
    args = parser.parse_args()
    if args.output is None:
        with TemporaryDirectory(prefix="pbz-examples-") as directory:
            generate(Path(directory))
        print("Temporary stores removed.")
    else:
        destination = args.output.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        generate(destination)
        print(f"Stores retained in {destination}")


if __name__ == "__main__":
    main()
