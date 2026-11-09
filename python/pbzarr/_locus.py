"""Lazy coordinates on a ``position`` dimension of one locus.

A ``region()`` view labels each row ``"contig:pos"`` and carries a numeric
``pos``. The values come from the contig name and a ``range`` of
contig-local positions, so a view of a whole contig builds no per-base
array until a caller asks for the values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from xarray import AlignmentError
from xarray.core.indexing import IndexSelResult
from xarray.indexes import Index, PandasIndex

DIM = "position"


def _take(positions, rows: np.ndarray) -> np.ndarray:
    if isinstance(positions, range):
        rows = np.where(rows < 0, rows + len(positions), rows)
        if rows.size and (rows.min() < 0 or rows.max() >= len(positions)):
            raise IndexError("row index out of range")
        return positions.start + rows.astype(np.int64) * positions.step
    return positions[rows]


def _as_range(positions: np.ndarray):
    if positions.size == 0:
        return range(0)
    start = int(positions[0])
    if np.array_equal(positions, np.arange(start, start + positions.size)):
        return range(start, start + positions.size)
    return positions


def _label(contig: str, position) -> str:
    return f"{contig}:{int(position)}"


class LocusValues:
    """1-D values of one locus: contig-local positions, or their labels.

    The values are made on demand. Slices stay lazy.
    """

    def __init__(self, contig: str, positions, labels: bool):
        self.contig = contig
        self.positions = positions
        self.labels = labels

    @property
    def shape(self) -> tuple[int]:
        return (len(self.positions),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(object) if self.labels else np.dtype(np.int64)

    def __len__(self) -> int:
        return len(self.positions)

    def _values(self, positions) -> np.ndarray:
        if self.labels:
            return np.array([_label(self.contig, p) for p in positions], dtype=object)
        if isinstance(positions, range):
            return np.arange(positions.start, positions.stop, positions.step, dtype=np.int64)
        return np.asarray(positions, dtype=np.int64)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = tuple(k for k in key if k is not Ellipsis) or (slice(None),)
            if len(key) != 1:
                raise IndexError("LocusValues is 1-D")
            (key,) = key
        if isinstance(key, slice):
            return type(self)(self.contig, self.positions[key], self.labels)
        if isinstance(key, (int, np.integer)):
            return self._values([_take(self.positions, np.asarray([key]))[0]])[0]
        rows = np.asarray(key)
        if rows.dtype == bool:
            (rows,) = np.nonzero(rows)
        return self._values(_take(self.positions, rows.ravel())).reshape(rows.shape)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values(self.positions), dtype=dtype)

    def __array_function__(self, func, types, args, kwargs):
        return func(*_materialize(args), **_materialize(kwargs))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return getattr(ufunc, method)(*_materialize(inputs), **_materialize(kwargs))

    def _repr_inline_(self, max_width: int) -> str:
        shown = self.positions if len(self.positions) <= 4 else [
            *self.positions[:2], None, *self.positions[-2:]
        ]
        items = [
            "..." if p is None else repr(_label(self.contig, p)) if self.labels else str(int(p))
            for p in shown
        ]
        return " ".join(items)[:max_width]

    def __repr__(self) -> str:
        return f"LocusValues({self._repr_inline_(80)})"


def _materialize(value):
    if isinstance(value, LocusValues):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_materialize(v) for v in value)
    if isinstance(value, dict):
        return {k: _materialize(v) for k, v in value.items()}
    return value


class LocusIndex(Index):
    """Index on ``position`` of one locus, with ``"contig:pos"`` labels.

    Two indexes align when they hold the same contig. Their rows match by
    position. Indexes of two contigs refuse to align.
    """

    def __init__(self, contig: str, positions):
        self.contig = contig
        self.positions = positions

    @classmethod
    def from_variables(cls, variables, *, options):
        ((name, var),) = variables.items()
        data = var._data
        if name != DIM or not isinstance(data, LocusValues):
            raise ValueError("a LocusIndex is made by region(), not by set_xindex")
        return cls(data.contig, data.positions)

    @classmethod
    def concat(cls, indexes, dim, positions=None):
        if positions is not None:
            raise NotImplementedError("LocusIndex concat with positions")
        if all(isinstance(i, LocusIndex) and i.contig == indexes[0].contig for i in indexes):
            joined = np.concatenate([i._positions_array() for i in indexes])
            return cls(indexes[0].contig, _as_range(joined))
        # Views of several contigs keep their labels in a plain index.
        return PandasIndex(pd.Index(np.concatenate([i.to_pandas_index() for i in indexes])), dim)

    def create_variables(self, variables=None):
        attrs = encoding = None
        if variables and DIM in variables:
            attrs, encoding = variables[DIM].attrs, variables[DIM].encoding
        values = LocusValues(self.contig, self.positions, labels=True)
        return {DIM: xr.Variable((DIM,), values, attrs=attrs, encoding=encoding)}

    def to_pandas_index(self) -> pd.Index:
        return pd.Index(np.asarray(LocusValues(self.contig, self.positions, True)), name=DIM)

    def isel(self, indexers):
        key = indexers[DIM]
        if isinstance(key, slice):
            return type(self)(self.contig, self.positions[key])
        if isinstance(key, xr.Variable):
            if key.dims != (DIM,):
                return None
            key = key.values
        rows = np.asarray(key)
        if rows.ndim != 1:
            return None
        if rows.dtype == bool:
            (rows,) = np.nonzero(rows)
        return type(self)(self.contig, _as_range(_take(self.positions, rows)))

    def _rows(self, labels) -> np.ndarray:
        wanted = np.asarray([self._position(label) for label in np.ravel(labels)], dtype=np.int64)
        rows = self._lookup(wanted)
        missing = rows < 0
        if missing.any():
            raise KeyError(f"not in this locus: {np.ravel(labels)[missing][:3].tolist()}")
        return rows.reshape(np.shape(labels))

    def _position(self, label) -> int:
        contig, _, position = str(label).rpartition(":")
        if contig != self.contig:
            raise KeyError(f"{label!r} is not on {self.contig!r}")
        return int(position)

    def _lookup(self, wanted: np.ndarray) -> np.ndarray:
        if isinstance(self.positions, range) and self.positions.step == 1:
            rows = wanted - self.positions.start
            return np.where((rows >= 0) & (rows < len(self.positions)), rows, -1)
        return pd.Index(np.asarray(_take(self.positions, np.arange(len(self.positions))))).get_indexer(
            wanted
        )

    def sel(self, labels, method=None, tolerance=None):
        if method is not None or tolerance is not None:
            raise NotImplementedError("selection by label supports exact labels only")
        key = labels[DIM]
        if isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("label slices take no step")
            start = 0 if key.start is None else int(self._rows([key.start])[0])
            stop = len(self.positions) if key.stop is None else int(self._rows([key.stop])[0]) + 1
            return IndexSelResult({DIM: slice(start, stop)})
        if isinstance(key, xr.Variable):
            key = key.values
        if np.ndim(key) == 0:
            return IndexSelResult({DIM: int(self._rows([key])[0])})
        return IndexSelResult({DIM: self._rows(key)})

    def _positions_array(self) -> np.ndarray:
        return np.asarray(_take(self.positions, np.arange(len(self.positions))))

    def equals(self, other, **kwargs) -> bool:
        if not isinstance(other, LocusIndex) or other.contig != self.contig:
            return False
        if isinstance(self.positions, range) and isinstance(other.positions, range):
            return self.positions == other.positions
        return np.array_equal(self._positions_array(), other._positions_array())

    def join(self, other, how="inner"):
        if not isinstance(other, LocusIndex) or other.contig != self.contig:
            raise AlignmentError(
                f"region() views of different contigs do not align: {self!r} and {other!r}"
            )
        a, b = self._positions_array(), other._positions_array()
        if how == "inner":
            joined = a[np.isin(a, b)]
        elif how == "outer":
            joined = np.union1d(a, b)
        elif how == "left":
            return self
        elif how == "right":
            return other
        else:
            raise ValueError(f"unsupported join {how!r}")
        return type(self)(self.contig, _as_range(joined))

    def reindex_like(self, other):
        if not isinstance(other, LocusIndex) or other.contig != self.contig:
            self.join(other)
        return {DIM: self._lookup(other._positions_array())}

    def _span(self) -> str:
        if isinstance(self.positions, range) and self.positions.step == 1:
            return f"{self.contig}:{self.positions.start}-{self.positions.stop}"
        return f"{self.contig}, {len(self.positions)} rows"

    def _repr_inline_(self, max_width: int) -> str:
        return f"LocusIndex({self._span()})"[:max_width]

    def __repr__(self) -> str:
        return f"LocusIndex({self._span()})"


def locus_coords(contig: str, start: int, stop: int) -> xr.Coordinates:
    """The ``position`` labels and the numeric ``pos`` of one locus, both lazy."""
    positions = range(start, stop)
    coords = xr.Coordinates.from_xindex(LocusIndex(contig, positions))
    return coords.assign(pos=xr.Variable((DIM,), LocusValues(contig, positions, labels=False)))
