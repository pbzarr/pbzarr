from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest
import xarray as xr
import zarr

import pbzarr

_REPO = Path(__file__).resolve().parents[2]


def _pbz() -> Path:
    binary = _REPO / "target" / "debug" / "pbz"
    if not binary.exists():
        if shutil.which("cargo") is None:
            pytest.skip("the pbz CLI is not built and cargo is not on PATH")
        subprocess.run(["cargo", "build", "-p", "pbz"], cwd=_REPO, check=True)
    return binary


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
    for command, labels in (("import", ["s1", "s2"]), ("append", ["s3", "s4"])):
        subprocess.run(
            [_pbz(), command, "d4", "-o", store, "--track", "depth", "--no-progress",
             *(f"{d4_fixture}:{label}" for label in labels)],
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
