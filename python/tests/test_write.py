from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest
import xarray as xr

import pbzarr
from _helpers import bed_fixture, pbz_binary, write_sizes


def _create(destination, *, reference):
    pbzarr.create(destination, reference=reference).close()


def test_create_opens_an_empty_store_and_refuses_an_existing_destination(tmp_path):
    destination = tmp_path / "empty.pbz"

    with pbzarr.create(destination, reference={"chr1": 1000}) as dataset:
        assert isinstance(dataset, xr.Dataset)
        assert not dataset.data_vars
        assert dict(pbzarr.Reference.from_xarray(dataset)) == {"chr1": 1000}

    with pytest.raises(FileExistsError):
        pbzarr.create(destination, reference={"chr1": 1000})


def test_import_d4_creates_a_new_store_from_the_source_headers(
    tmp_path, d4_fixture
):
    destination = tmp_path / "d4.pbz"

    report = pbzarr.import_d4(
        destination,
        "depth",
        (d4_fixture, "depth-a"),
        dim="replicate",
    )
    assert report["positions"] == 1000

    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["depth"].dims == ("position", "replicate")
        assert dataset["depth"].sizes["position"] == 1000
        assert dataset.coords["replicate"].values.tolist() == ["depth-a"]
        assert dataset["depth"].dtype == np.dtype("int32")
        expected = np.repeat((np.arange(100) % 50) + 1, 10)
        np.testing.assert_array_equal(dataset["depth"].values[:, 0], expected)
        assert dict(pbzarr.Reference.from_xarray(dataset)) == {"chr1": 1000}


def test_import_progress_writes_through_python_stderr(tmp_path, d4_fixture, capsys):
    pbzarr.import_d4(tmp_path / "d4.pbz", "depth", d4_fixture, progress=True)

    assert "depth: 100%" in capsys.readouterr().err


def test_import_adds_a_track_and_refuses_a_used_name(tmp_path, d4_fixture):
    destination = tmp_path / "d4.pbz"
    pbzarr.import_d4(destination, "depth", d4_fixture)
    pbzarr.import_d4(destination, "depth2", d4_fixture)

    with pytest.raises(pbzarr.PbzError, match="already has a track named"):
        pbzarr.import_d4(destination, "depth", d4_fixture)

    with pbzarr.open(destination, chunks=None) as dataset:
        assert set(dataset.data_vars) == {"depth", "depth2"}


def test_import_d4_cohort_opens_as_one_xarray_dataset(tmp_path, cohort_fixtures):
    destination = tmp_path / "cohort.pbz"
    _create(destination, reference={"chr1": 10000})

    report = pbzarr.import_d4(
        destination,
        "depth",
        list(zip(cohort_fixtures, ["s0", "s1"])),
    )
    assert report["positions"] == 10_000

    with xr.open_zarr(destination, consolidated=False) as dataset:
        assert set(dataset.data_vars) == {"depth"}
        assert dataset["depth"].dims == ("position", "sample")
        assert dataset["depth"].sizes["position"] == 10000
        assert dataset["sample"].values.tolist() == ["s0", "s1"]
        assert "pbz:reference" not in dataset.attrs
        assert dataset["contig_id"].dims == ("contig",)
        np.testing.assert_array_equal(dataset["contig_id"].values, ["chr1"])
        np.testing.assert_array_equal(dataset["contig_length"].values, [10000])
        expected = np.repeat((np.arange(1000) % 50) + 1, 10)
        np.testing.assert_array_equal(dataset["depth"].values[:, 0], expected)
        np.testing.assert_array_equal(dataset["depth"].values[:, 1], expected + 100)


def test_import_errors_name_a_missing_file_or_a_bad_argument(tmp_path, d4_fixture):
    destination = tmp_path / "d4.pbz"

    with pytest.raises(FileNotFoundError, match="missing.d4"):
        pbzarr.import_d4(destination, "depth", tmp_path / "missing.d4")
    with pytest.raises(FileNotFoundError):
        pbzarr.open(destination)
    with pytest.raises(ValueError, match="all provide labels or none"):
        pbzarr.import_d4(destination, "depth", [(d4_fixture, "a"), d4_fixture])
    assert not destination.exists()


def test_import_bigwig_path_list_defaults_to_sample_axis(tmp_path, bigwig_fixtures):
    first, second = bigwig_fixtures
    destination = tmp_path / "bigwig.pbz"
    _create(destination, reference={"chr1": 20})

    report = pbzarr.import_bigwig(destination, "signal", [first, second], threads=2)
    assert report["positions"] == 20

    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["signal"].dims == ("position", "sample")
        assert dataset.coords["sample"].values.tolist() == ["first", "second"]
        assert dataset["signal"].isel(position=0).values.tolist() == [1.0, 11.0]


def test_import_bigwig_fill_zero_declares_no_null(tmp_path, bigwig_fixtures):
    default = tmp_path / "default.pbz"
    zero = tmp_path / "zero.pbz"
    pbzarr.import_bigwig(default, "signal", bigwig_fixtures[0])
    pbzarr.import_bigwig(zero, "signal", bigwig_fixtures[0], fill=0)

    with pbzarr.open(default, chunks=None) as a, pbzarr.open(zero, chunks=None) as b:
        assert np.isnan(a["signal"].encoding["_FillValue"])
        assert "_FillValue" not in b["signal"].encoding
        assert b["signal"].values.tolist() == a["signal"].values.tolist()
    with pytest.raises(ValueError, match="fill"):
        pbzarr.import_bigwig(tmp_path / "bad.pbz", "signal", bigwig_fixtures[0], fill="zero")


def test_import_bed_gaps_are_nan_by_default_and_fill_gives_a_value(tmp_path):
    bed = bed_fixture("coverage")
    genome = write_sizes(tmp_path, "genome", [("chr1", 30)])

    pbzarr.import_bed(tmp_path / "na.pbz", bed, genome=genome, column="coverage")
    pbzarr.import_bed(tmp_path / "zero.pbz", bed, genome=genome, column="coverage", fill=0)

    with pbzarr.open(tmp_path / "na.pbz", chunks=None) as dataset:
        values = dataset["coverage"].values
        assert dataset["coverage"].encoding["dtype"] == np.dtype("uint8")
        assert values[:20].tolist() == [7] * 20
        assert np.isnan(values[20:]).all()
    with pbzarr.open(tmp_path / "zero.pbz", chunks=None) as dataset:
        assert dataset["coverage"].dtype == np.dtype("uint8")
        assert dataset["coverage"].values.tolist() == [7] * 20 + [0] * 10


def test_import_bed_schema_entries_give_track_dtype_and_fill(tmp_path):
    bed = bed_fixture("mixed")
    genome = write_sizes(tmp_path, "genome", [("chr1", 30)])
    destination = tmp_path / "schema.pbz"

    pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        schema={
            "coverage": {"track": "depth", "fill": 0},
            5: {"track": "score", "dtype": "float64"},
        },
    )

    with pbzarr.open(destination, chunks=None) as dataset:
        assert set(dataset.data_vars) == {"depth", "score"}
        assert dataset["depth"].values[-10:].tolist() == [0] * 10
        assert dataset["score"].dtype == np.dtype("float64")
        assert dataset["score"].values[0] == 1.5
        assert np.isnan(dataset["score"].values[-10:]).all()


def test_import_bed_single_column_is_scalar(tmp_path):
    bed = bed_fixture("coverage")
    destination = tmp_path / "bed.pbz"
    _create(destination, reference={"chr1": 20})

    report = pbzarr.import_bed(
        destination,
        bed,
        track="depth",
        column="coverage",
        dtype="int32",
    )
    assert report["positions"] == 20

    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["depth"].dims == ("position",)
        # An int32 track with a null decodes to float64.
        assert dataset["depth"].encoding["dtype"] == np.dtype("int32")
        assert dataset["depth"].dtype == np.dtype("float64")
        assert dataset["depth"].values.tolist() == [7] * 20


def test_import_bed_schema_maps_columns_and_infers_missing_dtypes(tmp_path):
    bed = bed_fixture("mixed")
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "multi.pbz"

    report = pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        schema={"coverage": "int32", "score": None, "is_max": None},
    )
    assert report["positions"] == 20

    with pbzarr.open(destination, chunks=None) as dataset:
        assert set(dataset.data_vars) == {"coverage", "score", "is_max"}
        assert dataset["coverage"].encoding["dtype"] == np.dtype("int32")
        assert dataset["score"].dtype == np.dtype("float32")
        assert dataset["is_max"].dtype == np.dtype("bool")
        assert dataset["score"].values[[0, -1]].tolist() == [1.5, -2.5]
        assert dataset["is_max"].values[[0, -1]].tolist() == [True, False]


def test_import_bed_refuses_coordinate_and_unnamed_numbered_columns(tmp_path):
    bed = bed_fixture("coverage")
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "numbered.pbz"

    with pytest.raises(ValueError, match="coordinate column"):
        pbzarr.import_bed(destination, bed, genome=genome, column=2, track="start")
    with pytest.raises(ValueError, match="needs a track name"):
        pbzarr.import_bed(destination, bed, genome=genome, column=4)

    pbzarr.import_bed(destination, bed, genome=genome, column=4, track="depth")
    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["depth"].values.tolist() == [7] * 20


def test_import_bed_wide_form_gathers_fields_into_one_track(tmp_path):
    bed = bed_fixture("wide")
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "wide.pbz"
    _create(destination, reference={"chr1": 20})

    pbzarr.import_bed(
        destination,
        bed,
        genome=genome,
        track="stats",
        dim="metric",
    )

    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["stats"].dims == ("position", "metric")
        assert dataset.coords["metric"].values.tolist() == ["cov", "mapq"]
        assert dataset["stats"].isel(position=0).values.tolist() == [4, 60]
        assert dataset["stats"].isel(position=-1).values.tolist() == [9, 300]


def test_import_bed_multi_source_matrix(tmp_path):
    first = bed_fixture("sample_a")
    second = bed_fixture("sample_b")
    genome = write_sizes(tmp_path, "genome", [("chr1", 20)])
    destination = tmp_path / "matrix.pbz"
    _create(destination, reference={"chr1": 20})

    pbzarr.import_bed(
        destination,
        [(first, "tumor"), (second, "normal")],
        genome=genome,
        dim="sample",
    )

    with pbzarr.open(destination, chunks=None) as dataset:
        assert dataset["depth"].dims == ("position", "sample")
        assert dataset.coords["sample"].values.tolist() == ["tumor", "normal"]
        assert dataset["depth"].isel(position=0).values.tolist() == [1, 2]


def test_import_d4_codecs_override_controls_zarr_metadata(tmp_path, d4_fixture):
    destination = tmp_path / "codecs.pbz"
    _create(destination, reference={"chr1": 1000})
    pbzarr.import_d4(
        destination,
        "depth",
        d4_fixture,
        codecs=[
            {"name": "bytes", "configuration": {"endian": "little"}},
            {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
        ],
    )

    meta = json.loads((destination / "depth" / "zarr.json").read_text())
    assert [c["name"] for c in meta["codecs"]] == ["bytes", "zstd"]

    with pytest.raises(ValueError, match="codecs conflicts with"):
        pbzarr.import_d4(
            destination,
            "depth2",
            d4_fixture,
            chunk_size=100,
            codecs=[{"name": "zstd", "configuration": {"level": 3, "checksum": False}}],
        )


def test_cli_reads_64_bit_integer_tracks_from_write(tmp_path):
    destination = tmp_path / "wide.pbz"
    big = 2**40
    signed = np.asarray([-big] * 4 + [7] * 6, dtype=np.int64)
    unsigned = np.asarray([big] * 4 + [0] * 6, dtype=np.uint64)
    with pbzarr.create(destination, reference={"chr1": 10}):
        pass
    pbzarr.write(destination, {
        "signed": xr.DataArray(signed, dims=("position",)),
        "unsigned": xr.DataArray(unsigned, dims=("position",)),
    }).close()

    def pbz(*args):
        return subprocess.run(
            [pbz_binary(), *args], check=True, capture_output=True, text=True
        ).stdout

    assert pbz("view", destination, "--track", "signed") == (
        f"#chrom\tstart\tend\tsigned\nchr1\t0\t4\t{-big}\nchr1\t4\t10\t7\n"
    )
    assert pbz("view", destination, "--track", "unsigned") == (
        f"#chrom\tstart\tend\tunsigned\nchr1\t0\t4\t{big}\nchr1\t4\t10\t0\n"
    )
    assert pbz("stat", destination, "--track", "signed", "-s", "hist") == (
        f"#value\tcount\n{-big}\t4\n7\t6\n"
    )
    assert pbz("stat", destination, "--track", "unsigned", "-s", "max") == (
        f"#chrom\tstart\tend\tmax\nchr1\t0\t10\t{big}\n"
    )
