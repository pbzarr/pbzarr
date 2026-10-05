"""Shared pytest fixtures."""
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.fixture
def d4_fixture() -> Path:
    """Committed .d4 file: chr1, length 1000, bands of 10, value (i % 50) + 1."""
    return Path(__file__).resolve().parents[2] / "fixtures" / "d4" / "banded_1k.d4"


@pytest.fixture
def cohort_fixtures() -> list[Path]:
    """Committed .d4 cohort: chr1, length 10000, value 100*s + (i % 50) + 1."""
    d4 = Path(__file__).resolve().parents[2] / "fixtures" / "d4"
    return [d4 / "cohort_s0.d4", d4 / "cohort_s1.d4"]


@pytest.fixture
def write_bigwig(tmp_path: Path):
    """Build a small .bw file with a known shape. Skip if pybigtools missing."""
    pybigtools = pytest.importorskip("pybigtools")

    def _make(chrom: str, length: int, sample: str = "A", base: float = 0.0) -> Path:
        out = tmp_path / f"{sample}.bw"
        intervals = [
            (chrom, i * 10, (i + 1) * 10, float((i % 50) + 1) + base)
            for i in range(length // 10)
        ]
        bw = pybigtools.open(str(out), "w")
        bw.write({chrom: length}, iter(intervals))
        bw.close()
        return out

    return _make
