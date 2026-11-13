"""The 0-d reference coordinate: alignment between genomes, survival, checks."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import xarray as xr

from pbzarr import Reference
from pbzarr._coord import NAME, ReferenceIndex, _check_full, attach, digest, same_reference

REF = Reference(["chr1", "chr2", "chrM"], [10, 7, 3])
REORDERED = Reference(["chr2", "chr1", "chrM"], [7, 10, 3])
N = 20
V = np.stack([np.arange(N), np.arange(N) * 10], 1).astype(np.float32) + 1


def track(reference: Reference = REF) -> xr.DataArray:
    da = xr.DataArray(
        V, dims=("position", "sample"), coords={"sample": ["a", "b"]}, name="signal"
    )
    return attach(da, reference)


def has_reference(obj) -> bool:
    return isinstance(obj.xindexes.get(NAME), ReferenceIndex)


def test_coordinate_value_is_the_digest():
    signal = track()
    assert signal[NAME].item() == digest(REF)
    assert "3 contigs" in repr(signal.xindexes[NAME])


@pytest.mark.parametrize(
    "combine",
    [
        lambda a, b: a + b,
        lambda a, b: xr.align(a, b, join="exact"),
        lambda a, b: xr.merge([a.rename("x"), b.rename("y")]),
        lambda a, b: xr.concat([a, b], "sample"),
        lambda a, b: xr.where(a > 5, a, b),
    ],
    ids=["add", "align", "merge", "concat", "where"],
)
def test_two_genomes_refuse_to_align(combine):
    with pytest.raises(xr.AlignmentError):
        combine(track(REF), track(REORDERED))


def test_equal_references_align():
    other = track(Reference(list(REF), list(REF.values())))
    result = track() + other
    assert has_reference(result)
    np.testing.assert_array_equal(result.values, V * 2)


@pytest.mark.parametrize(
    "op",
    [
        lambda s: s / s.sum("position") * 1e6,
        lambda s: s.mean("sample"),
        lambda s: s.where(s > 5),
        lambda s: xr.concat([s.sel(sample=["a"]), s.sel(sample=["b"])], "sample"),
        lambda s: xr.merge([s.rename("x"), (s * 2).rename("y")])["y"],
        lambda s: s.groupby("sample").mean(),
        lambda s: s.chunk(position=5).map_blocks(lambda b: b * 2).compute(),
        lambda s: pickle.loads(pickle.dumps(s)),
        lambda s: s.to_dataset(name="x")["x"],
    ],
    ids=[
        "arithmetic", "reduce", "where", "concat", "merge", "groupby",
        "map_blocks", "pickle", "to_dataset",
    ],
)
def test_coordinate_survives(op):
    result = op(track())
    assert has_reference(result)
    assert same_reference(Reference.from_xarray(result), REF)


def test_from_xarray():
    signal = track()
    assert Reference.from_xarray(signal) is REF
    assert Reference.from_xarray(signal.to_dataset()) is REF
    with pytest.raises(ValueError, match="no pbzarr reference"):
        Reference.from_xarray(signal.drop_vars(NAME))


def test_check_full():
    signal = track()
    _check_full(signal, REF)
    _check_full(signal.to_dataset(), REF)
    with pytest.raises(ValueError, match=r"has 12 rows but the reference has 20.*region\(\)"):
        _check_full(signal.isel(position=slice(0, 12)), REF)
