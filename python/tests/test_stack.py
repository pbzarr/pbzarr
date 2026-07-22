import pbzarr
from _helpers import htslib, write_bed_bgzip_tabix, write_sizes


def _single_sample(tmp_path, name, value):
    """A single-sample store with one scalar `coverage` track (constant value)."""
    bed = write_bed_bgzip_tabix(
        tmp_path, name, ["chrom", "start", "end", "coverage"], [("chr1", 0, 40, [str(value)])]
    )
    sizes = write_sizes(tmp_path, f"{name}_g", [("chr1", 40)])
    path = str(tmp_path / f"{name}.pbz")
    store = pbzarr.PbzStore.create(path)
    store.track("coverage").import_bed([str(bed)], column="coverage", dtype="int32", genome=str(sizes))
    return path


@htslib
def test_stack_single_sample_stores_into_cohort(tmp_path):
    s1 = _single_sample(tmp_path, "s1", 3)
    s2 = _single_sample(tmp_path, "s2", 8)

    cohort = pbzarr.stack([(s1, "s1"), (s2, "s2")], str(tmp_path / "cohort.pbz"))

    assert cohort.tracks() == ["coverage"]
    cov = cohort.track("coverage")
    assert cov.rank == 2
    assert cov.column_dim == "sample"
    assert cov.column_labels() == ["s1", "s2"]

    da = cov.region("chr1:0-40")
    assert da.shape == (40, 2)
    assert (da.sel(sample="s1").values == 3).all()
    assert (da.sel(sample="s2").values == 8).all()
