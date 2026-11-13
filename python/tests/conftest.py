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
def bigwig_fixtures() -> list[Path]:
    """Committed .bw files first and second: chr1, length 20, bands of 10,
    value (i % 50) + 1 + base with base 0 and 10."""
    bw = Path(__file__).resolve().parents[2] / "fixtures" / "bigwig"
    return [bw / "first.bw", bw / "second.bw"]
