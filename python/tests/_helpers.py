from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import zarr

PBZ_UUID = "d5bd7bea-0388-4e0a-9844-707c034e9add"
_REPO = Path(__file__).resolve().parents[2]


def pbz_binary() -> Path:
    """The debug pbz CLI. Build it when it is missing."""
    binary = _REPO / "target" / "debug" / "pbz"
    if not binary.exists():
        if shutil.which("cargo") is None:
            pytest.skip("the pbz CLI is not built and cargo is not on PATH")
        subprocess.run(["cargo", "build", "-p", "pbz"], cwd=_REPO, check=True)
    return binary


def write_pbz_store(store, reference: dict[str, int], name: str, values: np.ndarray,
                    *, dims: tuple[str, ...], chunks: tuple[int, ...],
                    shards: tuple[int, ...] | None = None,
                    labels: dict[str, list[str]] | None = None,
                    coordinates_attr: bool = True) -> None:
    """Write a format 0.5 store with one track, using zarr only."""
    group = zarr.open_group(store, mode="w", zarr_format=3)
    n = len(reference)
    contig = group.create_array(
        "contig_id", shape=(n,), chunks=(n,), dtype=str, dimension_names=("contig",)
    )
    contig[:] = np.asarray(list(reference))
    group.create_array(
        "contig_length",
        data=np.asarray(list(reference.values()), dtype=np.int64),
        chunks=(n,),
        dimension_names=("contig",),
    )
    track = group.create_array(
        name,
        shape=values.shape,
        chunks=chunks,
        shards=shards,
        dtype=values.dtype,
        fill_value=0,
        dimension_names=dims,
    )
    track[:] = values
    for dim, names in (labels or {}).items():
        label = group.create_array(
            dim, shape=(len(names),), chunks=(len(names),), dtype=str,
            dimension_names=(dim,),
        )
        label[:] = np.asarray(names)
    attrs = {"zarr_conventions": [{"uuid": PBZ_UUID, "name": "pbz"}], "pbz:version": "0.5"}
    if coordinates_attr:
        attrs["coordinates"] = "contig_id contig_length"
    group.attrs.update(attrs)


def write_bed_bgzip_tabix(dirp: Path, name: str, header: list[str],
                          rows: list[tuple[str, int, int, list[str]]]) -> Path:
    """Write a #-headed BED, bgzip it, tabix-index it. Rows must be sorted."""
    import pysam

    bed = dirp / f"{name}.bed"
    with bed.open("w") as f:
        f.write("#" + "\t".join(header) + "\n")
        for chrom, start, end, cells in rows:
            f.write("\t".join([chrom, str(start), str(end), *cells]) + "\n")
    gz = Path(pysam.tabix_index(str(bed), preset="bed", force=True))
    return gz


def write_sizes(dirp: Path, name: str, contigs: list[tuple[str, int]]) -> Path:
    """Write a 2-column chrom.sizes (accepted by Reference::from_fai)."""
    p = dirp / f"{name}.sizes"
    p.write_text("".join(f"{c}\t{n}\n" for c, n in contigs))
    return p
