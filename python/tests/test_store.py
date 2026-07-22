from pbzarr import PbzStore, Track
from _helpers import htslib, write_bed_bgzip_tabix, write_sizes, rust_fixture_store


def test_open_rust_fixture_lists_tracks(tmp_path):
    store = PbzStore(str(rust_fixture_store(tmp_path)))
    assert store.tracks() == ["depth", "mask"]
    assert isinstance(store.track("depth"), Track)
    tree = store.tree()
    assert set(tree.children) == {"depth", "mask"}


def test_open_non_store_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    import pytest
    with pytest.raises(Exception):
        PbzStore(str(tmp_path / "empty"))


@htslib
def test_create_import_read_roundtrip(tmp_path):
    p = str(tmp_path / "out.pbz")
    store = PbzStore.create(p)
    bed = write_bed_bgzip_tabix(tmp_path, "s1", ["chrom", "start", "end", "coverage"],
                                [("chr1", 0, 50, ["7"])])
    sizes = write_sizes(tmp_path, "g", [("chr1", 50)])
    store.track("coverage").import_bed([str(bed)], column="coverage", dtype="int32", genome=str(sizes))

    assert store.tracks() == ["coverage"]
    da = store.track("coverage").region("chr1:10-20")
    assert da.shape == (10,) and (da.values == 7).all()
