from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import pbzarr

pysam = pytest.importorskip("pysam")


def _write_bam(tmp_path, contig: str, length: int, reads: list[tuple[int, int]]):
    """Build a coordinate-sorted, indexed BAM with one plain `nM` read per
    `(start, length)` pair. Returns the sorted BAM path."""
    header = {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": contig, "LN": length}],
    }
    unsorted = tmp_path / "unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=header) as fh:
        for i, (start, read_len) in enumerate(reads):
            a = pysam.AlignedSegment()
            a.query_name = f"r{i}"
            a.query_sequence = "A" * read_len
            a.flag = 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigar = [(0, read_len)]  # nM
            a.query_qualities = pysam.qualitystring_to_array("I" * read_len)
            fh.write(a)

    sorted_path = tmp_path / "reads.bam"
    pysam.sort("-o", str(sorted_path), str(unsorted))
    pysam.index(str(sorted_path))
    return sorted_path


def test_import_bam_depth_single_source_matches_hand_computed_vector(tmp_path):
    contig, length = "chr1", 50
    # r0 covers [0,10), r1 covers [5,15), r2 covers [20,30).
    reads = [(0, 10), (5, 10), (20, 10)]
    bam = _write_bam(tmp_path, contig, length, reads)

    expected = np.zeros(length, dtype=np.int32)
    for start, read_len in reads:
        expected[start : start + read_len] += 1

    destination = tmp_path / "depth.pbz"
    pbzarr.create_store(destination)

    report = pbzarr.import_bam(destination, "depth", bam, mode="depth")
    assert report["contigs_written"] == 1
    # length=50 is one write-unit under the default 1,000,000-position chunk
    # size, and the reads give it data, so exactly one task runs, none skipped.
    assert report["tasks_completed"] == 1
    assert report["tasks_skipped"] == 0
    assert report["bytes_written"] > 0

    collection = pbzarr.open(destination, chunks=None)
    try:
        assert isinstance(collection, xr.DataTree)
        depth = collection["depth"].to_dataset(inherit=False)
        assert depth["values"].dims == ("position",)
        assert depth["values"].dtype == np.dtype("int32")
        assert depth["values"].values.tolist() == expected.tolist()
    finally:
        collection.close()
