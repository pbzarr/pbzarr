"""The 0-d ``reference`` coordinate that carries the genome through xarray."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import xarray as xr
from xarray import AlignmentError
from xarray.indexes import Index

from ._reference import Reference, bin_width

NAME = "reference"


def same_reference(a: Reference, b: Reference) -> bool:
    """Equal contig names and lengths, in order."""
    return a is b or (a.names == b.names and np.array_equal(a.lengths, b.lengths))


def digest(reference: Reference) -> str:
    """Short digest of the contig names and lengths, in order."""
    h = hashlib.sha1()
    for name, length in zip(reference.names, reference.lengths.tolist()):
        h.update(f"{name}\t{length}\n".encode())
    return h.hexdigest()[:8]


class ReferenceIndex(Index):
    """Index on one 0-d coordinate. It carries the Reference through xarray."""

    def __init__(self, reference: Reference, name: str = NAME):
        self.reference = reference
        self.name = name
        self.digest = digest(reference)

    @classmethod
    def from_variables(cls, variables, *, options):
        if len(variables) != 1:
            raise ValueError("a ReferenceIndex needs exactly one coordinate")
        ((name, var),) = variables.items()
        if var.ndim != 0:
            raise ValueError("a ReferenceIndex needs a 0-d coordinate")
        if "reference" not in options:
            raise ValueError("a ReferenceIndex needs the option reference=")
        return cls(options["reference"], name)

    def create_variables(self, variables=None):
        return {self.name: xr.Variable((), self.digest)}

    def to_pandas_index(self) -> pd.Index:
        # map_blocks and groupby call this.
        return pd.Index([self.digest], name=self.name)

    def equals(self, other, **kwargs) -> bool:
        return isinstance(other, ReferenceIndex) and same_reference(
            self.reference, other.reference
        )

    def join(self, other, how="inner"):
        raise AlignmentError(
            f"the objects use different references: {self!r} and {other!r}"
        )

    def reindex_like(self, other, **kwargs):
        if not self.equals(other):
            self.join(other)
        return {}

    def sel(self, labels, **kwargs):
        raise NotImplementedError(
            f"selection on {self.name!r} is not supported; use region() for one locus"
        )

    def rename(self, name_dict, dims_dict):
        if self.name not in name_dict:
            return self
        return type(self)(self.reference, name_dict[self.name])

    def _repr_inline_(self, max_width: int) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"ReferenceIndex({len(self.reference)} contigs, {self.digest})"


def attach(obj, reference: Reference):
    """Return ``obj`` with the ``reference`` coordinate of ``reference``."""
    return obj.assign_coords(xr.Coordinates.from_xindex(ReferenceIndex(reference)))


def reference_of(obj) -> Reference:
    index = obj.xindexes.get(NAME)
    if isinstance(index, ReferenceIndex):
        return index.reference
    if NAME in obj.coords:
        raise ValueError(
            f"the {NAME!r} coordinate has no ReferenceIndex (coarsen drops it); "
            "take the reference from the source array, or pass reference="
        )
    raise ValueError(
        "this object has no pbzarr reference; open it with pbzarr, or pass reference="
    )


def _check_full(array, reference: Reference) -> None:
    """Refuse a genomic dimension that does not span the whole reference."""
    for dim, size in array.sizes.items():
        if bin_width(dim) is None:
            continue
        expected = reference.size(dim)
        if size != expected:
            raise ValueError(
                f"{dim!r} has {size} rows but the reference has {expected}. "
                "This is part of the genome. "
                "Use the whole array, or region() for one locus."
            )
        if dim in array.xindexes:
            raise ValueError(
                f"{dim!r} has an index, so this is a region() view, not the whole "
                "genome. Use the whole array."
            )
