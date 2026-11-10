from __future__ import annotations

import subprocess

import numpy as np
import pytest
import xarray as xr

import pbzarr
from _helpers import pbz_binary


def _cli_import(store, sources, *flags) -> None:
    subprocess.run(
        [pbz_binary(), "import", "d4", "-o", store, "--track", "depth", "--no-progress",
         *flags, *sources],
        check=True, capture_output=True,
    )


@pytest.fixture
def stores(tmp_path, cohort_fixtures):
    sources = [f"{path}:s{index}" for index, path in enumerate(cohort_fixtures)]
    directory = tmp_path / "cohort.pbz"
    archive = tmp_path / "cohort.zip"
    _cli_import(directory, sources)
    _cli_import(archive, sources)
    return directory, archive


def test_open_reads_a_zip_store_like_a_directory_store(stores):
    directory, archive = stores
    with pbzarr.open(directory) as expected, pbzarr.open(archive) as actual:
        assert actual["reference"].item() == expected["reference"].item()
        xr.testing.assert_identical(actual["depth"].compute(), expected["depth"].compute())

        view = pbzarr.region(actual, "chr1:100-120")
        np.testing.assert_array_equal(
            view["depth"].values, pbzarr.region(expected, "chr1:100-120")["depth"].values
        )
        intervals = ["chr1:0-50", "chr1:5000-5100"]
        xr.testing.assert_identical(
            pbzarr.reduce_regions(actual, intervals, "mean").compute(),
            pbzarr.reduce_regions(expected, intervals, "mean").compute(),
        )


def test_write_and_import_refuse_a_zip_store(stores, d4_fixture):
    _, archive = stores
    with pbzarr.open(archive) as dataset:
        with pytest.raises(ValueError, match="write-once"):
            pbzarr.write(dataset, {"over_10": dataset["depth"] > 10})
    with pytest.raises(ValueError, match="write-once"):
        pbzarr.import_d4(archive.with_name("new.zip"), "depth", d4_fixture)


def test_open_refuses_a_missing_zip_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pbzarr.open(tmp_path / "missing.zip")
