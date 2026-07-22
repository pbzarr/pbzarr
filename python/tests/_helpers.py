import shutil
import subprocess
from pathlib import Path

import pytest

# Repo root: python/tests/_helpers.py -> tests -> python -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]


def _have(bin_: str) -> bool:
    return shutil.which(bin_) is not None


htslib = pytest.mark.skipif(
    not (_have("bgzip") and _have("tabix")),
    reason="bgzip/tabix (htslib) not on PATH",
)


def write_bed_bgzip_tabix(dirp: Path, name: str, header: list[str],
                          rows: list[tuple[str, int, int, list[str]]]) -> Path:
    """Write a #-headed BED, bgzip it, tabix-index it. Rows must be sorted."""
    bed = dirp / f"{name}.bed"
    with bed.open("w") as f:
        f.write("#" + "\t".join(header) + "\n")
        for chrom, start, end, cells in rows:
            f.write("\t".join([chrom, str(start), str(end), *cells]) + "\n")
    subprocess.run(["bgzip", "-f", str(bed)], check=True)
    gz = dirp / f"{name}.bed.gz"
    subprocess.run(["tabix", "-f", "-p", "bed", str(gz)], check=True)
    return gz


def write_sizes(dirp: Path, name: str, contigs: list[tuple[str, int]]) -> Path:
    """Write a 2-column chrom.sizes (accepted by Genome::from_fai)."""
    p = dirp / f"{name}.sizes"
    p.write_text("".join(f"{c}\t{n}\n" for c, n in contigs))
    return p


def rust_fixture_store(dirp: Path) -> Path:
    """Build the flat fixture store via the Rust example (mask 1D bool + depth 2D uint16).

    Runs cargo from the repo root so the workspace resolves regardless of pytest's cwd.
    """
    p = dirp / "fix.pbz"
    subprocess.run(
        ["cargo", "run", "-q", "-p", "pbzarr", "--example", "fixture_smoke_store", "--", str(p)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return p
