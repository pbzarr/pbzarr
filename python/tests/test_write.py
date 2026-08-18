from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pbzarr
from _helpers import htslib, write_bed_bgzip_tabix, write_sizes


def _track(path, name):
    collection = pbzarr.open(path, chunks=None)
    assert isinstance(collection, xr.DataTree)
    return collection, collection[name].to_dataset(inherit=False)


def test_create_store_returns_none_and_refuses_an_existing_destination(tmp_path):
    destination = tmp_path / "empty.pbz"

    assert pbzarr.create_store(destination) is None
    collection = pbzarr.open(destination)
    try:
        assert isinstance(collection, xr.DataTree)
        assert not collection.children
    finally:
        collection.close()

    with pytest.raises(FileExistsError):
        pbzarr.create_store(destination)


def test_import_d4_one_labeled_source_with_column_dim_is_two_dimensional(
    tmp_path, write_d4
):
    source = write_d4("chr1", 20, "depth")
    destination = tmp_path / "d4.pbz"

    with pytest.raises(pbzarr.PbzError):
        pbzarr.import_d4(tmp_path / "missing.pbz", "depth", source)

    pbzarr.create_store(destination)
    assert (
        pbzarr.import_d4(
            destination,
            "depth",
            (source, "depth-a"),
            column_dim="replicate",
        )
        is None
    )

    collection, depth = _track(destination, "depth")
    try:
        assert depth["values"].dims == ("position", "replicate")
        assert depth["replicate"].values.tolist() == ["depth-a"]
        assert depth["values"].dtype == np.dtype("int32")
        assert depth["values"].isel(position=0, replicate=0).item() == 1
    finally:
        collection.close()


def test_import_bigwig_path_list_defaults_to_sample_axis(tmp_path, write_bigwig):
    first = write_bigwig("chr1", 20, "first", base=0.0)
    second = write_bigwig("chr1", 20, "second", base=10.0)
    destination = tmp_path / "bigwig.pbz"
    pbzarr.create_store(destination)

    assert pbzarr.import_bigwig(destination, "signal", [first, second]) is None

    collection, signal = _track(destination, "signal")
    try:
        assert signal["values"].dims == ("position", "sample")
        assert signal["sample"].values.tolist() == ["first", "second"]
        assert signal["values"].isel(position=0).values.tolist() == [1.0, 11.0]
    finally:
        collection.close()


@htslib
def test_import_bed_one_path_is_scalar(tmp_path):
    bed = write_bed_bgzip_tabix(
        tmp_path,
        "single",
        ["chrom", "start", "end", "coverage"],
        [("chr1", 0, 20, ["7"])],
    )
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "bed.pbz"
    pbzarr.create_store(destination)

    assert (
        pbzarr.import_bed(
            destination,
            "coverage",
            bed,
            column="coverage",
            dtype="int32",
            genome=genome,
        )
        is None
    )

    collection, coverage = _track(destination, "coverage")
    try:
        assert coverage["values"].dims == ("position",)
        assert coverage["values"].values.tolist() == [7] * 20
    finally:
        collection.close()


@htslib
def test_import_bed_multi_writes_mapped_scalar_tracks(tmp_path):
    bed = write_bed_bgzip_tabix(
        tmp_path,
        "multi",
        ["chrom", "start", "end", "coverage", "score", "is_max"],
        [
            ("chr1", 0, 10, ["4", "1.5", "true"]),
            ("chr1", 10, 20, ["9", "-2.5", "false"]),
        ],
    )
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "multi.pbz"
    pbzarr.create_store(destination)

    assert (
        pbzarr.import_bed_multi(
            destination,
            bed,
            {"coverage": "int32", "score": "float32", "is_max": "bool"},
            genome=genome,
        )
        is None
    )

    collection = pbzarr.open(destination, chunks=None)
    try:
        assert set(collection.children) == {"coverage", "score", "is_max"}
        assert collection["coverage"]["values"].dtype == np.dtype("int32")
        assert collection["score"]["values"].values[[0, -1]].tolist() == [
            1.5,
            -2.5,
        ]
        assert collection["is_max"]["values"].values[[0, -1]].tolist() == [
            True,
            False,
        ]
    finally:
        collection.close()


@htslib
def test_stack_writes_labeled_collection_and_refuses_existing_destination(tmp_path):
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    sources = []
    for name, value in [("left", 3), ("right", 8)]:
        bed = write_bed_bgzip_tabix(
            tmp_path,
            name,
            ["chrom", "start", "end", "coverage"],
            [("chr1", 0, 20, [str(value)])],
        )
        source = tmp_path / f"{name}.pbz"
        pbzarr.create_store(source)
        pbzarr.import_bed(
            source,
            "coverage",
            bed,
            column="coverage",
            dtype="int32",
            genome=genome,
        )
        sources.append((source, name))

    destination = tmp_path / "cohort.pbz"
    assert pbzarr.stack(sources, destination) is None

    collection, coverage = _track(destination, "coverage")
    try:
        assert coverage["values"].dims == ("position", "sample")
        assert coverage["sample"].values.tolist() == ["left", "right"]
        assert coverage["values"].isel(position=0).values.tolist() == [3, 8]
    finally:
        collection.close()

    with pytest.raises(FileExistsError):
        pbzarr.stack(sources, destination)

    not_pbz = tmp_path / "not-pbz"
    not_pbz.mkdir()
    with pytest.raises(pbzarr.PbzError):
        pbzarr.stack(not_pbz, tmp_path / "invalid.pbz")


def test_import_d4_codecs_override_controls_zarr_metadata(tmp_path, write_d4):
    import json

    source = write_d4("chr1", 20, "depth")
    destination = tmp_path / "codecs.pbz"
    pbzarr.create_store(destination)
    pbzarr.import_d4(
        destination,
        "depth",
        source,
        codecs=[
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
        ],
    )

    meta = json.loads((destination / "depth" / "values" / "zarr.json").read_text())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"]

    with pytest.raises(pbzarr.PbzError):
        pbzarr.import_d4(
            destination,
            "depth2",
            source,
            chunk_size=100,
            codecs=[{"name": "zstd", "configuration": {"level": 3, "checksum": False}}],
        )
