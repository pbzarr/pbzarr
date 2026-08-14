"""Validate a PBZ store's consolidated metadata through released zarr-python 3.

Usage::

    pixi run python scripts/validate_consolidated.py <store> [<node-path> ...]

Opens the store with the default ``use_consolidated`` policy and asserts that
every listed node path is visible in the group's consolidated metadata, so a
reader never has to fetch a per-node ``zarr.json``. With no node paths given
the script prints what it found and exits 0.
"""
from pathlib import Path
import sys

import zarr


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    store = Path(argv[0])
    expected = argv[1:]

    group = zarr.open_group(store, mode="r")
    consolidated = group.metadata.consolidated_metadata
    assert consolidated is not None, f"{store}: root has no consolidated metadata"
    assert consolidated.kind == "inline", consolidated.kind
    assert consolidated.must_understand is False, consolidated.must_understand

    # ``metadata`` is the nested in-memory form (immediate children only);
    # ``flattened_metadata`` re-expands it to full paths from the root.
    found = set(consolidated.flattened_metadata)
    missing = [path for path in expected if path not in found]
    assert not missing, f"{store}: missing from consolidated metadata: {missing}"

    # The consolidated view alone must resolve every node, including the
    # multiscale level arrays.
    for path in sorted(found):
        node = group[path]
        if isinstance(node, zarr.Array):
            assert node.shape is not None, path

    print(f"consolidated metadata OK: {len(found)} nodes")
    for path in sorted(found):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
