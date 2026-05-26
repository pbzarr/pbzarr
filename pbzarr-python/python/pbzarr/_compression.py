"""Blosc(zstd-5, byte-shuffle) compression — matches the Rust pbzarr default."""
from __future__ import annotations


def default_data_codecs() -> list:
    """Return the codec list applied to every data array.

    Matches `pbzarr/src/store.rs::default_data_codecs` on the Rust side
    so cross-language reads decompress identically.
    """
    try:
        from zarr.codecs import BloscCodec, BloscShuffle
        return [BloscCodec(cname="zstd", clevel=5, shuffle=BloscShuffle.shuffle)]
    except ImportError:
        # Fallback to numcodecs path; both work with zarr-python v3.
        import numcodecs
        return [numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)]
