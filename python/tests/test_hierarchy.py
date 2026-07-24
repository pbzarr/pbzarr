"""Polymorphic open + the same-genome dataset() view."""
import dask.array
import numpy as np
import pytest

import pbzarr
from _helpers import rust_fixture_store


def test_open_dispatches_store_vs_track(tmp_path):
    p = rust_fixture_store(tmp_path)
    assert isinstance(pbzarr.open(str(p)), pbzarr.PbzStore)
    assert isinstance(pbzarr.open(f"{p}/depth"), pbzarr.Track)


def test_cross_kind_guards(tmp_path):
    p = rust_fixture_store(tmp_path)
    with pytest.raises(pbzarr.PbzError):
        pbzarr.PbzStore(f"{p}/depth")
    with pytest.raises(pbzarr.PbzError):
        pbzarr.Track.open(str(p))


def test_dataset_assembles_same_genome_view(tmp_path):
    ds = pbzarr.PbzStore(str(rust_fixture_store(tmp_path))).dataset()
    assert set(ds.data_vars) == {"mask", "depth"}
    assert ds["depth"].dims == ("position", "sample")
    assert ds["mask"].dims == ("position",)
    assert list(ds["sample"].values) == ["A", "B", "C"]
    # position is a shared axis with no index (never materialize a genome-length index)
    assert "position" not in ds.indexes
    # full genome travels with the Dataset
    assert list(ds["contigs"].values) == ["chr1", "chr2"]
    assert list(ds["offsets"].values) == [0, 2000, 3000]
    assert ds.attrs["genome_checksum"].startswith("md5:")
    assert ds.attrs["genome_name"] == "GRCh38"
    # a real value: fixture writes depth[i, 1] = i * 2
    assert ds["depth"].isel(position=5).sel(sample="B").compute().item() == 10


def test_dataset_backing_follows_store_chunks(tmp_path):
    p = rust_fixture_store(tmp_path)
    lazy = pbzarr.PbzStore(str(p), chunks={}).dataset()
    assert isinstance(lazy["depth"].data, dask.array.Array)
    eager = pbzarr.PbzStore(str(p), chunks=None).dataset()
    assert isinstance(eager["depth"].data, np.ndarray)
