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


@htslib
def test_import_bed_multi_creates_per_column_tracks(tmp_path):
    p = str(tmp_path / "multi.pbz")
    store = PbzStore.create(p)
    bed = write_bed_bgzip_tabix(
        tmp_path,
        "s1",
        ["chrom", "start", "end", "coverage", "score", "is_max"],
        [("chr1", 0, 20, ["4", "1.5", "true"]), ("chr1", 20, 40, ["9", "-2.5", "false"])],
    )
    sizes = write_sizes(tmp_path, "g", [("chr1", 40)])

    tracks = store.import_bed_multi(
        str(bed),
        {"coverage": "int32", "score": "float32", "is_max": "bool"},
        genome=str(sizes),
    )

    assert [t.name for t in tracks] == ["coverage", "score", "is_max"]
    assert store.tracks() == ["coverage", "is_max", "score"]  # sorted listing
    assert store.track("coverage").dtype == "int32"
    assert store.track("score").dtype == "float32"
    assert store.track("is_max").dtype == "bool"
    assert all(t.rank == 1 for t in tracks)

    cov = store.track("coverage").region("chr1:0-40")
    assert cov.values[:20].tolist() == [4] * 20 and cov.values[20:].tolist() == [9] * 20
    score = store.track("score").region("chr1:0-40")
    assert score.values[0] == 1.5 and score.values[-1] == -2.5
    is_max = store.track("is_max").region("chr1:0-40")
    assert bool(is_max.values[0]) is True and bool(is_max.values[-1]) is False
