"""Node-kind discrimination, mirroring the Rust `perbase:kind` marker.

Both a collection (store) root and a track group carry the same
`zarr_conventions` perbase marker; `perbase:kind` is what tells them apart.
Stores written before that key existed (0.4/0.5) are classified by inference.
"""
from __future__ import annotations


def is_perbase(attrs: dict) -> bool:
    return any(c.get("name") == "perbase" for c in attrs.get("zarr_conventions", []))


def kind_of(attrs: dict) -> str:
    """Return "track" or "collection". Falls back for pre-kind stores: a
    `perbase:version` (only tracks carried a `perbase:` block then) means a
    track; otherwise the bare-marker collection root."""
    k = attrs.get("perbase:kind")
    if k in ("track", "collection"):
        return k
    return "track" if "perbase:version" in attrs else "collection"
