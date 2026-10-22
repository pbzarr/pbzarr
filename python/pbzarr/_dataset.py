"""The PBZ store handle: one format 0.5 store, opened as one Xarray Dataset."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
import xarray as xr
import zarr

from . import _native
from ._native import PbzError
from ._reference import Reference, bin_width
from ._regions import _PBZ_SOURCE_ATTR, reduce_regions_dataset, regions_dataset

FORMAT_VERSION = "0.5"
_PBZ_UUID = "d5bd7bea-0388-4e0a-9844-707c034e9add"
_LEGACY_UUID = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c"
_STORED_CHUNKS = object()
_SUPPORTED_DTYPES = frozenset(
    {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
     "uint64", "float32", "float64"}
)
# Attributes Xarray or CF decoding acts on; a written track must not carry them.
_DECODING_ATTRS = frozenset(
    {"scale_factor", "add_offset", "missing_value", "_FillValue", "_Unsigned",
     "calendar", "dtype"}
)
# Encoding keys Xarray sets only when decoding changed the stored values.
_DECODED_KEYS = frozenset(
    {"scale_factor", "add_offset", "missing_value", "_Unsigned", "units", "calendar"}
)
_TIME_UNITS = frozenset(
    {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds", "nanoseconds"}
)


@dataclass(frozen=True)
class _Stored:
    """One array's stored identity, as listed from the store."""

    shape: tuple[int, ...]
    dtype: str
    dims: tuple[str, ...]
    token: str


class Dataset:
    """A handle on one PBZ store.

    The handle owns the store and its reference. Values are ordinary Xarray
    objects: ``ds["depth"]`` is a lazy ``xr.DataArray``. Select a region with
    ``ds["depth"].isel(position=ds.reference.rows("chr1:1000-2000"))``.
    """

    def __init__(self, source, *, mode, data, root_attrs, reference,
                 storage_options, chunks, selected, catalog) -> None:
        self._source = source
        self._mode = mode
        self._chunks = chunks
        self._selected = selected
        self._data = data
        self._root_attrs = root_attrs
        self._reference = reference
        self._storage_options = storage_options
        self._catalog = catalog

    @property
    def reference(self) -> Reference:
        return self._reference

    @property
    def tracks(self) -> Mapping[str, xr.DataArray]:
        return dict(self._data.data_vars)

    @property
    def coords(self) -> Mapping[str, xr.DataArray]:
        return dict(self._data.coords)

    @property
    def sizes(self) -> Mapping[str, int]:
        return dict(self._data.sizes)

    @property
    def attrs(self) -> Mapping[str, object]:
        return dict(self._data.attrs)

    def __getitem__(self, name: str) -> xr.DataArray:
        if name not in self._data.data_vars:
            raise KeyError(f"unknown track {name!r}")
        return self._data[name]

    def __contains__(self, name: object) -> bool:
        return name in self._data.data_vars

    def __iter__(self) -> Iterator[str]:
        return iter(self._data.data_vars)

    def __repr__(self) -> str:
        return f"<pbzarr.Dataset mode={self._mode!r}>\n{self._data!r}"

    def to_xarray(self) -> xr.Dataset:
        """The opened store as an ordinary Xarray Dataset."""
        return self._data

    def close(self) -> None:
        self._data.close()

    def __enter__(self) -> Dataset:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def regions(self, intervals) -> xr.Dataset:
        """Raw values of several regions, packed along ``position``.

        The result has a ``region`` dimension that keeps each interval's
        contig, start and stop. Tracks without a leading ``position``
        dimension are skipped.
        """
        return regions_dataset(_position_tracks(self._data), self._reference, intervals)

    def reduce_regions(self, intervals, reducer: str, /, **kwargs) -> xr.Dataset:
        """One statistic per region and track, on a ``region`` dimension."""
        return reduce_regions_dataset(
            _position_tracks(self._data), self._reference, intervals, reducer, **kwargs
        )

    def write(self, tracks: Mapping[str, xr.DataArray], *, encoding=None) -> None:
        """Add new tracks to the store.

        Each value is a plain ``xr.DataArray``. A genomic dimension
        (``position`` or ``bin_N``) must come first and span the reference.
        Shared non-genomic dimensions must match the store's labels. If the
        write fails, the arrays it created are removed; it is not atomic
        against a crash.
        """
        if self._mode != "r+":
            raise PbzError("write() needs a store opened with mode='r+'")
        if _catalog(self._source, self._storage_options) != self._catalog:
            raise PbzError("store changed since open; reopen it before writing")
        if not isinstance(tracks, Mapping) or not tracks:
            raise TypeError("tracks must be a nonempty mapping of name to DataArray")
        encoding = dict(encoding or {})

        variables = {}
        coords = {}
        for name, value in tracks.items():
            variable, new_coords = self._check_track(name, value)
            variables[name] = variable
            for dim, (_, labels) in new_coords.items():
                if dim in coords and not np.array_equal(coords[dim][1], labels):
                    raise ValueError(f"tracks give different labels for new dimension {dim!r}")
            coords.update(new_coords)
        for name in variables:
            if name in coords:
                raise ValueError(f"{name!r} is both a track and a new dimension")
        unknown = set(encoding) - set(variables)
        if unknown:
            raise ValueError(f"encoding names tracks not being written: {sorted(unknown)}")

        new = xr.Dataset(variables, coords=coords, attrs=self._root_attrs)
        var_encoding = {}
        for name in variables:
            entry = dict(encoding.get(name, {}))
            entry.setdefault("_FillValue", None)
            var_encoding[name] = entry
        try:
            new.to_zarr(
                self._source,
                mode="a",
                consolidated=False,
                encoding=var_encoding,
                storage_options=_options(self._storage_options),
            )
        except BaseException:
            self._remove([*variables, *coords])
            raise
        self._reopen(list(variables))

    def _remove(self, names: list[str]) -> None:
        group = zarr.open_group(
            self._source, mode="r+", use_consolidated=False,
            **_storage_kwargs(self._storage_options),
        )
        for name in names:
            if name not in self._catalog and name in group:
                del group[name]

    def _check_track(self, name, value) -> tuple[xr.Variable, dict]:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"track name must be a nonempty string without '/': {name!r}")
        if bin_width(name) is not None:
            raise ValueError(f"{name!r} is a genomic dimension name and cannot name a track")
        stored_dims = {dim for entry in self._catalog.values() for dim in entry.dims}
        if name in self._catalog or name in stored_dims:
            raise ValueError(f"{name!r} already exists in the store")
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"track {name!r} must be an xarray DataArray")
        if np.dtype(value.dtype).name not in _SUPPORTED_DTYPES:
            raise TypeError(f"track {name!r} has unsupported dtype {value.dtype}")

        for axis, dim in enumerate(value.dims):
            if bin_width(dim) is None:
                continue
            if dim in self._catalog:
                raise ValueError(f"track {name!r}: {dim!r} is already an array in the store")
            if axis != 0:
                raise ValueError(f"track {name!r}: genomic dimension {dim!r} must come first")
            expected = self._reference.size(dim)
            if value.sizes[dim] != expected:
                raise ValueError(
                    f"{name!r} has {value.sizes[dim]} rows on {dim!r}; "
                    f"the reference has {expected}"
                )

        new_coords = {}
        for dim in value.dims[1:] if value.ndim and bin_width(value.dims[0]) else value.dims:
            labels = value.coords[dim].values if dim in value.coords else None
            stored = self._catalog.get(dim)
            if stored is not None:
                if value.sizes[dim] != stored.shape[0]:
                    raise ValueError(
                        f"track {name!r}: {dim!r} has {value.sizes[dim]} labels; "
                        f"the store has {stored.shape[0]}"
                    )
                if labels is not None and not np.array_equal(labels, self._labels(dim)):
                    raise ValueError(f"track {name!r}: {dim!r} labels differ from the store's")
                continue
            if dim in stored_dims:
                raise ValueError(f"track {name!r}: {dim!r} has no label coordinate in the store")
            if labels is None:
                raise ValueError(f"track {name!r}: new dimension {dim!r} needs a coordinate")
            if any(label is None for label in labels.tolist()) or (
                labels.dtype.kind == "f" and np.isnan(labels).any()
            ):
                raise ValueError(f"track {name!r}: {dim!r} labels must not be missing")
            if len(set(labels.tolist())) != labels.size:
                raise ValueError(f"track {name!r}: {dim!r} labels must be unique")
            if labels.dtype.kind == "U":
                labels = labels.astype(object)
            new_coords[dim] = (dim, labels)

        # pbz:summary and the zarrs provenance note describe the source array,
        # and Xarray keeps attributes through arithmetic.
        attrs = {k: v for k, v in value.attrs.items() if k not in {"pbz:summary", "_zarrs"}}
        for key in attrs:
            if key.startswith("pbz:") or key in {"zarr_conventions", "coordinates"}:
                raise ValueError(f"track {name!r}: attribute {key!r} is reserved")
            units = attrs[key] if key == "units" else None
            if key in _DECODING_ATTRS or (
                isinstance(units, str) and (" since " in units or units in _TIME_UNITS)
            ):
                raise ValueError(
                    f"track {name!r}: attribute {key!r} changes decoding; "
                    "pass _FillValue through encoding= instead"
                )
        try:
            json.dumps(attrs)
        except TypeError as error:
            raise TypeError(f"track {name!r}: attributes must be JSON-serializable") from error
        return xr.Variable(value.dims, value.data, attrs=attrs), new_coords

    def _labels(self, dim: str) -> np.ndarray:
        if dim in self._data.coords:
            return self._data[dim].values
        group = zarr.open_group(
            self._source, mode="r", use_consolidated=False,
            **_storage_kwargs(self._storage_options),
        )
        return np.asarray(group[dim][:])

    def _reopen(self, written: list[str]) -> None:
        selected = None if self._selected is None else [*self._selected, *written]
        self._data.close()
        reopened = open(
            self._source,
            mode=self._mode,
            tracks=selected,
            chunks=self._chunks,
            storage_options=self._storage_options,
        )
        self._data = reopened._data
        self._root_attrs = reopened._root_attrs
        self._catalog = reopened._catalog
        self._selected = selected


def open(
    source,
    *,
    mode: str = "r",
    tracks: Sequence[str] | None = None,
    chunks=_STORED_CHUNKS,
    storage_options: Mapping[str, object] | None = None,
) -> Dataset:
    """Open a PBZ store as a ``pbzarr.Dataset`` handle.

    ``mode="r"`` is read-only; ``mode="r+"`` allows ``write``. ``tracks=``
    limits which tracks the handle exposes. By default each track is a Dask
    array chunked by its stored shard (or chunk) shape; ``chunks=None``
    gives lazy NumPy-backed arrays, and any other value goes to
    ``xr.open_zarr``.
    """
    if mode not in {"r", "r+"}:
        raise ValueError("mode must be 'r' or 'r+'")
    if isinstance(source, (str, os.PathLike)):
        source = os.fspath(source)
    root_attrs = _root_attrs(source, storage_options)
    _check_identity(root_attrs)
    catalog = _catalog(source, storage_options)

    data = xr.open_zarr(
        source,
        chunks=None if chunks is _STORED_CHUNKS else chunks,
        consolidated=False,
        storage_options=_options(storage_options),
    )
    try:
        reference = _reference(data, root_attrs)
        _check_genomic_dims(data, reference)
        if chunks is _STORED_CHUNKS:
            data = data.assign(
                {name: _stored_chunks(value) for name, value in data.data_vars.items()}
            )
        for name, value in data.data_vars.items():
            _attach_source(value, source, name, catalog[name], storage_options)
        selected = None
        if tracks is not None:
            selected = [tracks] if isinstance(tracks, str) else list(tracks)
            data = _select(data, selected)
    except Exception:
        data.close()
        raise
    return Dataset(
        source,
        mode=mode,
        data=data,
        root_attrs=root_attrs,
        reference=reference,
        storage_options=storage_options,
        chunks=chunks,
        selected=selected,
        catalog=catalog,
    )


def create(
    destination,
    *,
    reference: Mapping[str, int],
) -> Dataset:
    """Create an empty PBZ store and open it with ``mode="r+"``.

    ``reference`` is the ordered mapping of contig name to length that
    every track shares. The destination must not exist.
    """
    path = os.fspath(destination)
    if not path:
        raise ValueError("destination must not be empty")
    if os.path.lexists(path):
        raise FileExistsError(f"destination already exists: {path}")
    if not isinstance(reference, Mapping) or not reference:
        raise TypeError("reference must be a nonempty mapping of contig name to length")
    contigs = []
    for name, length in reference.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"contig name must be a nonempty string; got {name!r}")
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 0:
            raise ValueError(f"length of contig {name!r} must be a nonnegative integer")
        contigs.append((name, int(length)))
    _native.create_store(path, contigs)
    return open(path, mode="r+")


def _options(storage_options):
    return None if storage_options is None else dict(storage_options)


def _storage_kwargs(storage_options) -> dict:
    return {} if storage_options is None else {"storage_options": dict(storage_options)}


def _root_attrs(source, storage_options) -> dict:
    group = zarr.open_group(
        source, mode="r", use_consolidated=False, **_storage_kwargs(storage_options)
    )
    return dict(group.attrs)


def _catalog(source, storage_options) -> dict[str, _Stored]:
    group = zarr.open_group(
        source, mode="r", use_consolidated=False, **_storage_kwargs(storage_options)
    )
    catalog = {}
    for name, array in group.arrays():
        metadata = array.metadata.to_dict()
        token = hashlib.sha1(
            json.dumps(metadata, sort_keys=True, default=str).encode()
        ).hexdigest()
        catalog[name] = _Stored(
            shape=tuple(array.shape),
            dtype=str(array.dtype),
            dims=tuple(metadata.get("dimension_names") or ()),
            token=token,
        )
    return catalog


def _check_identity(attrs: Mapping) -> None:
    conventions = attrs.get("zarr_conventions")
    uuids = [
        entry.get("uuid")
        for entry in conventions if isinstance(entry, Mapping)
    ] if isinstance(conventions, list) else []
    matches = uuids.count(_PBZ_UUID)
    if matches == 0:
        if _LEGACY_UUID in uuids:
            raise PbzError(
                "this is a format 0.4 PBZ store (per-track groups); this version "
                "of pbzarr reads format 0.5 only"
            )
        raise PbzError("not a PBZ store: no pbz entry in zarr_conventions")
    if matches > 1:
        raise PbzError("invalid PBZ store: zarr_conventions lists pbz more than once")
    version = attrs.get("pbz:version")
    if version != FORMAT_VERSION:
        raise PbzError(f"unsupported pbz:version {version!r}; expected {FORMAT_VERSION!r}")


def _reference(data: xr.Dataset, attrs: Mapping) -> Reference:
    pointers = attrs.get("pbz:reference")
    if not isinstance(pointers, Mapping):
        raise PbzError("invalid PBZ store: pbz:reference is missing")
    names_key, lengths_key = pointers.get("names"), pointers.get("lengths")
    for key in (names_key, lengths_key):
        if not isinstance(key, str) or key not in data.variables:
            raise PbzError(f"invalid PBZ store: reference array {key!r} is missing")
    names = data[names_key].values.tolist()
    return Reference(names, data[lengths_key].values)


def _check_genomic_dims(data: xr.Dataset, reference: Reference) -> None:
    for dim, size in data.sizes.items():
        if bin_width(dim) is None:
            continue
        if dim in data.variables:
            raise PbzError(f"invalid PBZ store: genomic dimension {dim!r} has a stored coordinate")
        if size != reference.size(dim):
            raise PbzError(
                f"invalid PBZ store: {dim!r} has {size} rows; the reference spans "
                f"{reference.size(dim)}"
            )


def _stored_chunks(value: xr.DataArray) -> xr.DataArray:
    shape = value.encoding.get("shards") or value.encoding.get("chunks")
    if shape is None:
        return value
    return value.chunk(dict(zip(value.dims, shape, strict=True)))


def _attach_source(
    value: xr.DataArray, source, name: str, stored: _Stored, storage_options
) -> None:
    """Let region tasks reopen the raw array, only when decoding is the identity."""
    if value.chunks is None or not isinstance(source, str):
        return
    try:
        if np.dtype(stored.dtype) != value.dtype:
            return
    except TypeError:
        return
    if any(key in value.encoding for key in _DECODED_KEYS):
        return
    fill = value.encoding.get("_FillValue")
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
        return
    location = source if "://" in source else os.path.abspath(source)
    setattr(
        value.data,
        _PBZ_SOURCE_ATTR,
        {
            "source": location,
            "array_path": name,
            "storage_options": _options(storage_options),
            "token": stored.token,
        },
    )


def _select(data: xr.Dataset, names: list[str]) -> xr.Dataset:
    unknown = [name for name in names if name not in data.data_vars]
    if unknown:
        raise PbzError(f"unknown track {unknown[0]!r}")
    return data.drop_vars([name for name in data.data_vars if name not in names])


def _position_tracks(data: xr.Dataset) -> xr.Dataset:
    keep = []
    for name, value in data.data_vars.items():
        if not value.ndim or value.dims[0] != "position":
            continue
        if value.ndim > 2:
            raise PbzError(
                f"track {name!r} has {value.ndim} dimensions; region operations "
                "handle position plus at most one more"
            )
        keep.append(name)
    return data[keep]
