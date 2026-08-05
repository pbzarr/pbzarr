from pbzarr import PbzStore, Track
from _helpers import rust_fixture_store


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
