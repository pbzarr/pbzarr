"""Shared pytest fixtures."""
from __future__ import annotations
from pathlib import Path
import subprocess
import pytest


@pytest.fixture
def write_d4(tmp_path: Path):
    """Build a small .d4 file with a known shape. Skip if d4tools missing."""
    if subprocess.run(["which", "d4tools"], capture_output=True).returncode != 0:
        pytest.skip("d4tools not on PATH")

    def _make(chrom: str, length: int, sample: str = "A") -> Path:
        bg = tmp_path / f"{sample}.bg"
        sizes = tmp_path / f"{sample}.sizes"
        lines = []
        for i in range(length // 10):
            s, e = i * 10, (i + 1) * 10
            v = (i % 50) + 1
            lines.append(f"{chrom}\t{s}\t{e}\t{v}\n")
        bg.write_text("".join(lines))
        sizes.write_text(f"{chrom}\t{length}\n")
        out = tmp_path / f"{sample}.d4"
        rc = subprocess.run(
            ["d4tools", "create", "-z", "-g", str(sizes), str(bg), str(out)],
            capture_output=True,
        ).returncode
        assert rc == 0
        return out

    return _make
