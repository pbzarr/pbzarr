from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import xarray as xr
import zarr

import pbzarr
from _helpers import pbz_binary


def _check_copy(path: Path) -> None:
    root = json.loads((path / "zarr.json").read_text())
    copy = root["consolidated_metadata"]
    assert copy["kind"] == "inline"
    assert copy["must_understand"] is False
    arrays = {child.name for child in path.iterdir() if (child / "zarr.json").exists()}
    assert set(copy["metadata"]) == arrays
    with xr.open_zarr(path, consolidated=True) as dataset:
        assert set(dataset.variables) == arrays


def test_cli_import_and_append_write_the_copy(tmp_path, d4_fixture):
    store = tmp_path / "cohort.pbz"
    steps = (("import", ["s1", "s2"], ["--shard-columns", "1"]), ("append", ["s3", "s4"], []))
    for command, labels, layout in steps:
        subprocess.run(
            [pbz_binary(), command, "d4", store, "--track", "depth", "--no-progress",
             *layout, *(f"{d4_fixture}:{label}" for label in labels)],
            check=True, capture_output=True,
        )
        _check_copy(store)
    with xr.open_zarr(store, consolidated=True) as dataset:
        assert dataset["sample"].values.tolist() == ["s1", "s2", "s3", "s4"]


def test_python_write_writes_the_copy(tmp_path, d4_fixture):
    store = tmp_path / "d4.pbz"
    pbzarr.import_d4(store, "depth", d4_fixture)
    with pbzarr.open(store) as dataset:
        pbzarr.write(store, {"over_10": dataset["depth"] > 10}).close()
    _check_copy(store)
    assert "over_10" in json.loads((store / "zarr.json").read_text())[
        "consolidated_metadata"]["metadata"]


def test_open_reads_the_copy_and_write_lists_the_store(tmp_path, d4_fixture):
    store = tmp_path / "d4.pbz"
    pbzarr.import_d4(store, "depth", d4_fixture)
    group = zarr.open_group(store, mode="r+", use_consolidated=False)
    group.create_array("hidden", shape=(1000,), dtype="int32", dimension_names=["position"])

    with pbzarr.open(store) as dataset:
        assert "hidden" not in dataset.data_vars
        with pytest.raises(ValueError, match="already exists"):
            pbzarr.write(store, {"hidden": dataset["depth"]})
