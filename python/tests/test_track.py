from pbzarr import Track
from _helpers import rust_fixture_store


def test_track_metadata_from_rust_fixture(tmp_path):
    store = rust_fixture_store(tmp_path)
    depth = Track(str(store), "depth")
    assert depth.rank == 2
    assert depth.column_dim == "sample"
    assert depth.dtype == "uint16"
    assert depth.column_labels() == ["A", "B", "C"]
    assert depth.genome() == [("chr1", 2000), ("chr2", 1000)]
    assert depth.total_len() == 3000
    mask = Track(str(store), "mask")
    assert mask.rank == 1 and mask.column_dim is None and mask.column_labels() is None
