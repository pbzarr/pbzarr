"""Region strings parse the same way in Python and in the pbz CLI."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
import xarray as xr

import pbzarr

from _helpers import pbz_binary, write_pbz_store

REFERENCE = {"chr1": 3000, "HLA-A*01:01:01": 10, "HLA-A*01:01:01:01": 6}
ACCEPTED = {
    "chr1": ("chr1", 0, 3000),
    "chr1:1,000-2,000": ("chr1", 1000, 2000),
    " chr1:2500 ": ("chr1", 2500, 3000),
    "HLA-A*01:01:01:01": ("HLA-A*01:01:01:01", 0, 6),
    "HLA-A*01:01:01:2-5": ("HLA-A*01:01:01", 2, 5),
}
REFUSED = ["chr1:-2000", "chr1:1000-", "chr1:１０-２０", ":1-2", "chr1:9-3"]


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "regions.pbz"
    values = np.zeros(sum(REFERENCE.values()), dtype=np.int32)
    write_pbz_store(path, REFERENCE, "depth", values, dims=("position",), chunks=(1024,))
    return path


def _cli_bounds(store, text: str):
    run = subprocess.run(
        [pbz_binary(), "stat", store, "--track", "depth", "-r", text],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        return None
    contig, start, end = run.stdout.splitlines()[1].split("\t")[:3]
    return contig, int(start), int(end)


def test_python_and_cli_accept_and_refuse_the_same_strings(store) -> None:
    with pbzarr.open(store) as ds:
        for text, expected in ACCEPTED.items():
            view = pbzarr.region(ds["depth"], text)
            got = (view.region_contig.item(), view.region_start.item(), view.region_end.item())
            assert got == expected, text
            assert _cli_bounds(store, text) == expected, text
        for text in REFUSED:
            with pytest.raises(ValueError):
                pbzarr.region(ds["depth"], text)
            assert _cli_bounds(store, text) is None, text


def test_parse_region_keeps_open_ends_and_strips_commas() -> None:
    assert pbzarr.parse_region("chr1:1,000") == pbzarr.RegionQuery("chr1", 1000, None)
    assert pbzarr.parse_region("chr1:") == pbzarr.RegionQuery("chr1")
    with pytest.raises(ValueError):
        pbzarr.parse_region("chr1:1000-")


def test_region_matches_a_whole_contig_name_with_colons() -> None:
    reference = pbzarr.Reference(list(REFERENCE), list(REFERENCE.values()))
    genome = xr.DataArray(np.arange(reference.size()), dims="position")
    view = pbzarr.region(genome, "HLA-A*01:01:01:01", reference=reference)
    np.testing.assert_array_equal(view.values, np.arange(3010, 3016))
    assert view.region_contig.item() == "HLA-A*01:01:01:01"
