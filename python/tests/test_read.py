import numpy as np

from pbzarr import _read
from pbzarr._region import RegionQuery
from _helpers import rust_fixture_store


def test_read_scalar_region(tmp_path):
    store = rust_fixture_store(tmp_path)
    # 'mask' is 1D bool, contigs chr1(2000)/chr2(1000), offsets [0,2000,3000].
    da = _read.read_region(str(store), "mask", RegionQuery("chr2", 0, 10))
    assert da.dims == ("position",)
    assert da.shape == (10,)
    assert list(da["position"].values) == list(range(10))
    assert str(da["contig"].values) == "chr2"


def test_read_two_d_region_and_column(tmp_path):
    store = rust_fixture_store(tmp_path)
    # 'depth' is 2D uint16, col_dim 'sample', labels A/B/C.
    da = _read.read_region(str(store), "depth", RegionQuery("chr1", 5, 15))
    assert da.dims == ("position", "sample")
    assert da.shape == (10, 3)
    assert list(da["sample"].values) == ["A", "B", "C"]
    one = _read.read_region(str(store), "depth", RegionQuery("chr1", 5, 15), column="B")
    assert one.dims == ("position",)
    assert one.shape == (10,)


def test_gather_tags_region_coord(tmp_path):
    store = rust_fixture_store(tmp_path)
    rqs = [RegionQuery("chr1", 0, 10), RegionQuery("chr2", 0, 5)]
    da = _read.gather_regions(str(store), "mask", rqs)
    assert da.sizes["position"] == 15
    assert list(np.unique(da["region"].values)) == [0, 1]
    assert (da["region"].values[:10] == 0).all()
    assert (da["region"].values[10:] == 1).all()


def test_region_blocks_raw(tmp_path):
    store = rust_fixture_store(tmp_path)
    rqs = [RegionQuery("chr1", 0, 8), RegionQuery("chr2", 0, 4)]
    rb = _read.region_blocks(str(store), "depth", rqs)
    assert [b.shape for b in rb.blocks] == [(8, 3), (4, 3)]
    assert list(rb.columns) == ["A", "B", "C"]
    assert rb.regions == [("chr1", 0, 8), ("chr2", 0, 4)]
