import numpy as np

from pbzarr import Track
from pbzarr._native import create_store
from _helpers import htslib, write_bed_bgzip_tabix, write_sizes, rust_fixture_store


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


@htslib
def test_import_bed_scalar_then_read(tmp_path):
    p = str(tmp_path / "out.pbz")
    create_store(p)
    bed = write_bed_bgzip_tabix(
        tmp_path, "s1", ["chrom", "start", "end", "coverage"],
        [("chr1", 0, 30, ["4"]), ("chr1", 30, 60, ["9"])],
    )
    sizes = write_sizes(tmp_path, "g", [("chr1", 60)])
    t = Track(p, "coverage")
    t.import_bed([str(bed)], column="coverage", dtype="int32", genome=str(sizes))

    assert t.rank == 1 and t.dtype == "int32"
    da = t.region("chr1:0-60")
    assert da.dims == ("position",) and da.shape == (60,)
    assert (da.values[:30] == 4).all() and (da.values[30:] == 9).all()


@htslib
def test_import_bed_cohort_gather_reduce(tmp_path):
    p = str(tmp_path / "c.pbz")
    create_store(p)
    s1 = write_bed_bgzip_tabix(tmp_path, "s1", ["chrom", "start", "end", "coverage"], [("chr1", 0, 40, ["2"])])
    s2 = write_bed_bgzip_tabix(tmp_path, "s2", ["chrom", "start", "end", "coverage"], [("chr1", 0, 40, ["6"])])
    sizes = write_sizes(tmp_path, "g", [("chr1", 40)])
    t = Track(p, "coverage")
    t.import_bed([(str(s1), "s1"), (str(s2), "s2")], column="coverage", dtype="int32", genome=str(sizes))

    assert t.rank == 2 and t.column_dim == "sample" and t.column_labels() == ["s1", "s2"]
    means = t.region_reduced([("chr1", 0, 10), ("chr1", 20, 30)], reduce="mean")
    # each region is constant per sample: s1=2, s2=6
    assert means.sel(sample="s1").values.tolist() == [2.0, 2.0]
    assert means.sel(sample="s2").values.tolist() == [6.0, 6.0]
    rb = t.region_blocks([("chr1", 0, 5)])
    assert rb.blocks[0].shape == (5, 2)
