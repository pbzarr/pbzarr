"""Throwaway Dataset/Track API experiment; not part of the pbzarr package.

Question: how much domain code is necessary around real Xarray objects?
Numerical methods delegate; the wrappers own alignment, context and write intent.
"""

from __future__ import annotations

import operator
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import xarray as xr

from .coarsen import GenomicCoarsen
from .genomic_index import GenomicIndex, Reference, attach_indexes, get_index
from .storage import create_store, read_store, write_tracks


def _genomic_dims(array):
    return [d for d in array.dims if d == "position" or d.startswith("bin_")]


def _labels(array):
    """The prototype requires ordinary unique dimension labels."""
    for dim in array.dims:
        if dim in _genomic_dims(array):
            continue
        if dim not in array.coords or array.coords[dim].dims != (dim,):
            raise ValueError(f"Dimension {dim!r} needs a same-named coordinate")
        labels = array.get_index(dim)
        if not labels.is_unique or labels.hasnans:
            raise ValueError(f"Dimension {dim!r} needs unique, nonmissing labels")


def _compatible(left, right):
    """Reject implicit joins before Xarray gets a chance to perform them."""
    li, ri = get_index(left), get_index(right)
    if li is not None and ri is not None and not li.equals(ri):
        raise ValueError("Genomic indexes differ; explicitly align in Xarray first")
    for dim in set(left.dims) & set(right.dims):
        if dim in _genomic_dims(left):
            continue
        if not left.get_index(dim).equals(right.get_index(dim)):
            raise ValueError(f"Labels or order differ on {dim!r}")


def _select(array, region, labels):
    unknown = set(labels) - set(array.dims)
    if unknown:
        raise ValueError(f"Unknown dimensions: {sorted(unknown)}")
    if set(labels) & set(_genomic_dims(array)):
        raise ValueError("Use region= for genomic selection")
    result = array
    if region is not None:
        index = get_index(result)
        if index is None:
            raise ValueError("This Track has no genomic dimension")
        result = result.sel({index.dim: region})
    # Keeping selected axes makes the result useful for subsequent writes.
    for dim, value in labels.items():
        if isinstance(value, (str, int, float)):
            value = [value]
        result = result.sel({dim: value})
    _labels(result)
    return result


class Track:
    """A DataArray with an explicit domain boundary, not another array engine."""

    def __init__(self, array: xr.DataArray, *, reference=None, owner=None):
        if not isinstance(array, xr.DataArray):
            raise TypeError("Use Track.from_array for raw arrays")
        index = get_index(array)
        if index is not None and reference is not None and index.reference != reference:
            raise ValueError("The supplied reference and genomic index disagree")
        _labels(array)
        if index is not None:
            array = array.transpose(index.dim, ...)
        self._array = array.copy(deep=False)
        self._reference = index.reference if index is not None else reference
        self._owners = owner if isinstance(owner, tuple) else (() if owner is None else (owner,))

    @classmethod
    def from_array(cls, data, *, dims, coords=None, reference=None, attrs=None):
        """Declare raw genomic input to be in canonical full-reference order."""
        if isinstance(reference, Mapping):
            reference = Reference.from_mapping(reference)
        array = xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)
        if _genomic_dims(array):
            if reference is None:
                raise ValueError("Full genomic input requires reference=")
            if len(_genomic_dims(array)) != 1:
                raise ValueError("Only one genomic dimension is supported")
            dim = _genomic_dims(array)[0]
            index = GenomicIndex(reference, dim)
            if array.sizes[dim] != len(index.rows):
                raise ValueError("Input extent does not match the reference")
            array = array.assign_coords(xr.Coordinates.from_xindex(index))
        return cls(array, reference=reference)

    @classmethod
    def from_xarray(cls, array, *, reference=None):
        """Indexed genomic arrays carry their own context; never infer it by shape."""
        return cls(array, reference=reference)

    def _check_open(self):
        for owner in self._owners:
            owner._check_open()

    @property
    def dims(self):
        return self._array.dims

    @property
    def shape(self):
        return self._array.shape

    @property
    def sizes(self):
        return self._array.sizes

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def attrs(self):
        return MappingProxyType(self._array.attrs)

    @property
    def coords(self):
        return MappingProxyType({name: coord.copy(deep=False) for name, coord in self._array.coords.items()})

    @property
    def reference(self):
        return self._reference

    @property
    def index(self):
        return get_index(self._array)

    def to_xarray(self):
        self._check_open()
        return self._array.copy(deep=False)

    def _wrap(self, result):
        return Track(result, reference=self.reference, owner=self._owners)

    def select(self, *, region=None, **labels):
        self._check_open()
        return self._wrap(_select(self._array, region, labels))

    def _reduce(self, method, dim=None, **kwargs):
        self._check_open()
        kwargs.setdefault("keep_attrs", False)
        return self._wrap(getattr(self._array, method)(dim=dim, **kwargs))

    def sum(self, dim=None, **kwargs):
        return self._reduce("sum", dim, **kwargs)

    def mean(self, dim=None, **kwargs):
        return self._reduce("mean", dim, **kwargs)

    def min(self, dim=None, **kwargs):
        return self._reduce("min", dim, **kwargs)

    def max(self, dim=None, **kwargs):
        return self._reduce("max", dim, **kwargs)

    def any(self, dim=None, **kwargs):
        return self._reduce("any", dim, **kwargs)

    def all(self, dim=None, **kwargs):
        return self._reduce("all", dim, **kwargs)

    def _binary(self, other, operation, *, reflected=False):
        self._check_open()
        owners = self._owners
        if isinstance(other, Track):
            owners += other._owners
            other = other.to_xarray()
        if isinstance(other, xr.DataArray):
            _labels(other)
            _compatible(self._array, other)
        elif not isinstance(other, (int, float, bool, complex)):
            raise TypeError("Give nonscalar operands named dimensions first")
        with xr.set_options(arithmetic_join="exact", keep_attrs=False):
            result = operation(other, self._array) if reflected else operation(self._array, other)
        return Track(result, reference=self.reference, owner=tuple(dict.fromkeys(owners)))

    def __ge__(self, other):
        return self._binary(other, operator.ge)

    def __gt__(self, other):
        return self._binary(other, operator.gt)

    def __le__(self, other):
        return self._binary(other, operator.le)

    def __lt__(self, other):
        return self._binary(other, operator.lt)

    def __eq__(self, other):
        return self._binary(other, operator.eq)

    def __ne__(self, other):
        return self._binary(other, operator.ne)

    def __add__(self, other):
        return self._binary(other, operator.add)

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    def __truediv__(self, other):
        return self._binary(other, operator.truediv)

    def __rsub__(self, other):
        return self._binary(other, operator.sub, reflected=True)

    def __rtruediv__(self, other):
        return self._binary(other, operator.truediv, reflected=True)

    __radd__ = __add__
    __rmul__ = __mul__

    def __and__(self, other):
        return self._binary(other, operator.and_)

    def __or__(self, other):
        return self._binary(other, operator.or_)

    def __invert__(self):
        self._check_open()
        result = ~self._array
        result.attrs = {}
        return self._wrap(result)

    def __bool__(self):
        raise TypeError("Use .any() or .all(), then .compute().item()")

    def astype(self, dtype, **kwargs):
        self._check_open()
        return self._wrap(self._array.astype(dtype, **kwargs))

    def with_attrs(self, **attrs):
        self._check_open()
        return self._wrap(self._array.assign_attrs(attrs))

    def coarsen(self, *, position):
        self._check_open()
        return _TrackCoarsen(self, GenomicCoarsen(self._array, position=position))

    def compute(self, **kwargs):
        self._check_open()
        return Track(self._array.compute(**kwargs), reference=self.reference)

    def item(self):
        self._check_open()
        # Private probe-only check: .data can read a non-Dask storage backend.
        if self._array.ndim or not self._array.variable._in_memory:
            raise ValueError("item() requires a materialized scalar")
        return self._array.item()

    def __repr__(self):
        state = "chunked" if self._array.chunks is not None else "unchunked"
        index = self.index
        positions = f"{index.dim}:{index.rows}" if index is not None else "none"
        return f"Track(dims={self.dims}, shape={self.shape}, dtype={self.dtype}, {state}, index={positions})"


class _TrackCoarsen:
    """Five return-type adapters around the shared native coarsening helper."""

    def __init__(self, track, native):
        self._track, self._native = track, native

    def _call(self, name, *args, **kwargs):
        self._track._check_open()
        return self._track._wrap(getattr(self._native, name)(*args, **kwargs))

    def mean(self, **kwargs):
        return self._call("mean", **kwargs)

    def sum(self, **kwargs):
        return self._call("sum", **kwargs)

    def min(self, **kwargs):
        return self._call("min", **kwargs)

    def max(self, **kwargs):
        return self._call("max", **kwargs)

    def reduce(self, func, **kwargs):
        return self._call("reduce", func, **kwargs)


class Dataset:
    """A whole labeled dataset; only writable roots have mutation authority."""

    def __init__(self, native, reference, *, path=None, mode="r", owner=None):
        self._native, self.reference = native, reference
        self._path, self._mode = path, mode
        self._owners = owner if isinstance(owner, tuple) else (() if owner is None else (owner,))
        self._closed = False
        self._closers = [native.close] if not self._owners else []
        self._chunks = "auto"

    @classmethod
    def from_xarray(cls, native):
        pointers = native.attrs.get("pbz:reference", {})
        if not {"names", "lengths"} <= pointers.keys():
            raise ValueError("Dataset must declare pbz:reference")
        reference = Reference(
            tuple(native[pointers["names"]].values.tolist()),
            tuple(native[pointers["lengths"]].values.tolist()),
        )
        return cls(attach_indexes(native, reference), reference)

    @classmethod
    def open(cls, source, *, mode="r", chunks="auto"):
        if mode not in {"r", "r+"}:
            raise ValueError("Use mode='r' or 'r+'; create() makes new stores")
        native = read_store(source, chunks=chunks)
        result = cls.from_xarray(native)
        result._path, result._mode, result._chunks = Path(source), mode, chunks
        result._closers = [native.close]
        return result

    @classmethod
    def create(cls, destination, *, reference, coords=None, attrs=None):
        if isinstance(reference, Mapping):
            reference = Reference.from_mapping(reference)
        create_store(destination, reference, coords=coords, attrs=attrs)
        return cls.open(destination, mode="r+")

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Dataset is closed")
        for owner in self._owners:
            owner._check_open()

    @property
    def tracks(self):
        return MappingProxyType({name: self[name] for name in self._native.data_vars})

    @property
    def coords(self):
        return MappingProxyType({name: coord.copy(deep=False) for name, coord in self._native.coords.items()})

    @property
    def attrs(self):
        return MappingProxyType(self._native.attrs)

    @property
    def sizes(self):
        return self._native.sizes

    def __getitem__(self, name):
        self._check_open()
        if name not in self._native.data_vars:
            raise KeyError(f"{name!r} is not a track; coordinates live in .coords")
        return Track(self._native[name], reference=self.reference, owner=self)

    def to_xarray(self):
        self._check_open()
        result = self._native.copy(deep=False)
        if any(index.rows != range(self.reference.size(index.width))
               for index in result.xindexes.values() if isinstance(index, GenomicIndex)):
            result.attrs = {k: v for k, v in result.attrs.items()
                            if k not in {"zarr_conventions", "pbz:version"}}
        return result

    def select(self, *, region=None, **labels):
        self._check_open()
        ref_dim = self._native.attrs["pbz:reference"]["names"]
        if ref_dim in labels or set(labels) & set(_genomic_dims(self._native)):
            raise ValueError("Use region= to select genomic data; reference stays whole")
        unknown = set(labels) - set(self._native.dims)
        if unknown:
            raise ValueError(f"Unknown dimensions: {sorted(unknown)}")
        native = self._native
        if region is not None:
            for dim in _genomic_dims(native):
                native = native.sel({dim: region})
        for dim, value in labels.items():
            if isinstance(value, (str, int, float)):
                value = [value]
            native = native.sel({dim: value})
        _labels(native)
        return Dataset(native, self.reference, owner=self)

    def assign(self, **tracks):
        """In-memory composition only. Shared axes must already match exactly."""
        self._check_open()
        native = self._native
        owners = [self]
        for name, value in tracks.items():
            if name in native.coords or name in native.dims:
                raise ValueError(f"Track name {name!r} collides with a coordinate/dimension")
            if isinstance(value, Track):
                owners.extend(value._owners)
            array = value.to_xarray() if isinstance(value, Track) else value
            track = Track.from_xarray(array, reference=self.reference)
            index = track.index
            if index is not None:
                existing = native.xindexes.get(index.dim)
                if existing is not None and not index.equals(existing):
                    raise ValueError("assign() cannot scatter a selected genomic result")
            for dim in array.dims:
                if dim in native.dims and dim not in _genomic_dims(array):
                    if not array.get_index(dim).equals(native.get_index(dim)):
                        raise ValueError(f"Labels differ on {dim!r}; assign does not align")
            native = native.assign({name: track.to_xarray()})
        return Dataset(native, self.reference, owner=tuple(dict.fromkeys(owners)))

    def write(self, tracks, *, encoding=None):
        self._check_open()
        if self._path is None or self._mode != "r+" or self._owners:
            raise ValueError("Write through a writable root Dataset")
        arrays = {}
        for name, value in tracks.items():
            array = value.to_xarray() if isinstance(value, Track) else value
            arrays[name] = Track.from_xarray(array, reference=self.reference).to_xarray()
        write_tracks(self._path, self._native, arrays, encoding=encoding)
        native = read_store(self._path, chunks=self._chunks)
        self._closers.append(native.close)
        self._native = attach_indexes(native, self.reference)

    def compute(self, **kwargs):
        self._check_open()
        native = self._native.compute(**kwargs)
        native.set_close(None)
        return Dataset(native, self.reference)

    def close(self):
        if not self._closed:
            for close in self._closers:
                close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        rows = [f"Dataset(reference={dict(zip(self.reference.names, self.reference.lengths))}, tracks={len(self._native.data_vars)})"]
        for name in self._native.data_vars:
            rows.append(f"  {name}: {self[name]!r}")
        return "\n".join(rows)
