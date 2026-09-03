from pathlib import Path

import pysam


def write_bed_bgzip_tabix(dirp: Path, name: str, header: list[str],
                          rows: list[tuple[str, int, int, list[str]]]) -> Path:
    """Write a #-headed BED, bgzip it, tabix-index it. Rows must be sorted."""
    bed = dirp / f"{name}.bed"
    with bed.open("w") as f:
        f.write("#" + "\t".join(header) + "\n")
        for chrom, start, end, cells in rows:
            f.write("\t".join([chrom, str(start), str(end), *cells]) + "\n")
    gz = Path(pysam.tabix_index(str(bed), preset="bed", force=True))
    return gz


def write_sizes(dirp: Path, name: str, contigs: list[tuple[str, int]]) -> Path:
    """Write a 2-column chrom.sizes (accepted by Genome::from_fai)."""
    p = dirp / f"{name}.sizes"
    p.write_text("".join(f"{c}\t{n}\n" for c, n in contigs))
    return p
