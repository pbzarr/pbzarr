from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pbzarr
from _helpers import write_bed_bgzip_tabix, write_sizes


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
    tmp_path, d4_fixture
):
    destination = tmp_path / "d4.pbz"

    with pytest.raises(pbzarr.PbzError):
        pbzarr.import_d4(tmp_path / "missing.pbz", "depth", d4_fixture)

    pbzarr.create_store(destination)
    report = pbzarr.import_d4(
        destination,
        "depth",
        (d4_fixture, "depth-a"),
        column_dim="replicate",
    )
    assert report["contigs_written"] == 1

    collection, depth = _track(destination, "depth")
    try:
        assert depth["values"].dims == ("position", "replicate")
        assert depth["values"].sizes["position"] == 1000
        assert depth["replicate"].values.tolist() == ["depth-a"]
        assert depth["values"].dtype == np.dtype("int32")
        expected = np.repeat((np.arange(100) % 50) + 1, 10)
        np.testing.assert_array_equal(depth["values"].values[:, 0], expected)
    finally:
        collection.close()


def test_import_bigwig_path_list_defaults_to_sample_axis(tmp_path, write_bigwig):
    first = write_bigwig("chr1", 20, "first", base=0.0)
    second = write_bigwig("chr1", 20, "second", base=10.0)
    destination = tmp_path / "bigwig.pbz"
    pbzarr.create_store(destination)

    report = pbzarr.import_bigwig(destination, "signal", [first, second])
    assert report["contigs_written"] == 1

    collection, signal = _track(destination, "signal")
    try:
        assert signal["values"].dims == ("position", "sample")
        assert signal["sample"].values.tolist() == ["first", "second"]
        assert signal["values"].isel(position=0).values.tolist() == [1.0, 11.0]
    finally:
        collection.close()


def test_import_bed_single_column_is_scalar(tmp_path):
    bed = write_bed_bgzip_tabix(
        tmp_path,
        "single",
        ["chrom", "start", "end", "coverage"],
        [("chr1", 0, 20, ["7"])],
    )
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "bed.pbz"
    pbzarr.create_store(destination)

    report = pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        track="depth",
        column="coverage",
        dtype="int32",
        scales=[16],
    )
    assert report["contigs_written"] == 1
    assert (destination / "depth" / "scales" / "16" / "mean" / "zarr.json").exists()

    collection, depth = _track(destination, "depth")
    try:
        assert depth["values"].dims == ("position",)
        assert depth["values"].dtype == np.dtype("int32")
        assert depth["values"].values.tolist() == [7] * 20
    finally:
        collection.close()


def test_import_bed_schema_maps_columns_and_infers_missing_dtypes(tmp_path):
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

    report = pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        schema={"coverage": "int32", "score": None, "is_max": None},
    )
    assert report["contigs_written"] == 1

    collection = pbzarr.open(destination, chunks=None)
    try:
        assert set(collection.children) == {"coverage", "score", "is_max"}
        assert collection["coverage"]["values"].dtype == np.dtype("int32")
        assert collection["score"]["values"].dtype == np.dtype("float32")
        assert collection["is_max"]["values"].dtype == np.dtype("bool")
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


def test_import_bed_wide_form_gathers_fields_into_one_track(tmp_path):
    bed = write_bed_bgzip_tabix(
        tmp_path,
        "wide",
        ["chrom", "start", "end", "cov", "mapq"],
        [("chr1", 0, 10, ["4", "60"]), ("chr1", 10, 20, ["9", "300"])],
    )
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "wide.pbz"
    pbzarr.create_store(destination)

    pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        track="stats",
        column_dim="metric",
    )

    collection, stats = _track(destination, "stats")
    try:
        assert stats["values"].dims == ("position", "metric")
        assert stats["metric"].values.tolist() == ["cov", "mapq"]
        assert stats["values"].isel(position=0).values.tolist() == [4, 60]
        assert stats["values"].isel(position=-1).values.tolist() == [9, 300]
    finally:
        collection.close()


def test_import_bed_multi_source_matrix(tmp_path):
    first = write_bed_bgzip_tabix(
        tmp_path,
        "s1",
        ["chrom", "start", "end", "depth"],
        [("chr1", 0, 20, ["1"])],
    )
    second = write_bed_bgzip_tabix(
        tmp_path,
        "s2",
        ["chrom", "start", "end", "depth"],
        [("chr1", 0, 20, ["2"])],
    )
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "matrix.pbz"
    pbzarr.create_store(destination)

    pbzarr.import_bed(
        destination,
        [(first, "tumor"), (second, "normal")],
        genome=genome,
        column_dim="sample",
    )

    collection, depth = _track(destination, "depth")
    try:
        assert depth["values"].dims == ("position", "sample")
        assert depth["sample"].values.tolist() == ["tumor", "normal"]
        assert depth["values"].isel(position=0).values.tolist() == [1, 2]
    finally:
        collection.close()


def test_import_bed_rejects_a_two_path_tuple(tmp_path):
    with pytest.raises(ValueError, match="pass a list"):
        pbzarr.import_bed(
            tmp_path / "trap.pbz",
            ("a.bed.gz", "b.bed.gz"),
            genome=tmp_path / "genome.sizes",
        )


def test_import_d4_codecs_override_controls_zarr_metadata(tmp_path, d4_fixture):
    import json

    destination = tmp_path / "codecs.pbz"
    pbzarr.create_store(destination)
    pbzarr.import_d4(
        destination,
        "depth",
        d4_fixture,
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
            d4_fixture,
            chunk_size=100,
            codecs=[{"name": "zstd", "configuration": {"level": 3, "checksum": False}}],
        )
