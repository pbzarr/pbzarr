"""Open, create, and write a format 0.5 PBZ store as one Xarray Dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
import xarray as xr
from xarray import AlignmentError
import zarr

from . import _coord, _native
from ._native import PbzError
from ._reference import Reference, bin_width
from ._regions import _PBZ_SOURCE_ATTR

FORMAT_VERSION = "0.5"
_PBZ_UUID = "d5bd7bea-0388-4e0a-9844-707c034e9add"
_LEGACY_UUID = "b7e3c1a2-5f4d-4e8a-9c1b-2d3e4f5a6b7c"
_STORED_CHUNKS = object()
# Root arrays that are coordinates by name, whatever the coordinates attribute says.
_REFERENCE_ARRAYS = ("contig_id", "contig_length", "contig_md5")
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


class _Store:
    """The metadata of one store, read when a call starts."""

    def __init__(self, source, *, data, root_attrs, reference, storage_options, catalog) -> None:
        self.source = source
        self.data = data
        self.root_attrs = root_attrs
        self.reference = reference
        self.storage_options = storage_options
        self.catalog = catalog

    def remove(self, names: list[str]) -> None:
        group = zarr.open_group(
            self.source, mode="r+", use_consolidated=False,
            **_storage_kwargs(self.storage_options),
        )
        for name in names:
            if name not in self.catalog and name in group:
                del group[name]

    def check(self, name, value) -> tuple[xr.Variable, dict[str, xr.Variable]]:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"track name must be a nonempty string without '/': {name!r}")
        if bin_width(name) is not None:
            raise ValueError(f"{name!r} is a genomic dimension name and cannot name a track")
        if name == _coord.NAME:
            raise ValueError(f"{name!r} is the in-memory reference coordinate, not a track name")
        stored_dims = {dim for entry in self.catalog.values() for dim in entry.dims}
        if name in self.catalog or name in stored_dims:
            raise ValueError(f"{name!r} already exists in the store")
        if not isinstance(value, xr.DataArray):
            raise TypeError(f"track {name!r} must be an xarray DataArray")
        value = self._drop_reference(name, value)

        for axis, dim in enumerate(value.dims):
            if dim == _coord.NAME:
                raise ValueError(f"track {name!r}: {dim!r} cannot name a dimension")
            if bin_width(dim) is None:
                continue
            if dim in self.catalog:
                raise ValueError(f"track {name!r}: {dim!r} is already an array in the store")
            if axis != 0:
                raise ValueError(f"track {name!r}: genomic dimension {dim!r} must come first")
            expected = self.reference.size(dim)
            if value.sizes[dim] != expected:
                raise ValueError(
                    f"{name!r} has {value.sizes[dim]} rows on {dim!r}; "
                    f"the reference has {expected}"
                )
        genomic = bool(value.ndim) and bin_width(value.dims[0]) is not None
        data = _checked_data(f"track {name!r}", value, strings=not genomic)

        new_coords = {}
        for dim in value.dims[1:] if genomic else value.dims:
            labels = None
            if dim in value.coords:
                if value.coords[dim].dims != (dim,):
                    raise ValueError(
                        f"track {name!r}: the label coordinate {dim!r} lies on "
                        f"{value.coords[dim].dims}; it must lie on ({dim!r},)"
                    )
                labels = value.coords[dim].values
            stored = self.catalog.get(dim)
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
            new_coords[dim] = xr.Variable((dim,), labels)

        for coord_name, coord in value.coords.items():
            where = f"track {name!r}: coordinate {coord_name!r}"
            if any(bin_width(dim) is not None for dim in coord.dims):
                raise ValueError(
                    f"{where} lies on a genomic dimension; the format stores no "
                    "coordinate over position or bin_N. Drop it before write"
                )
            if coord_name in value.dims:
                continue
            in_store = coord_name in self.catalog or coord_name in stored_dims
            stored = self.catalog.get(coord_name)
            if coord.ndim == 0 and in_store and (stored is None or stored.dims):
                # What isel or sel leaves of a stored dimension, such as sample.
                continue
            if bin_width(coord_name) is not None:
                raise ValueError(f"{where} has a genomic dimension name")
            if in_store:
                if (
                    stored is None
                    or stored.dims != coord.dims
                    or not np.array_equal(coord.values, self._labels(coord_name))
                ):
                    raise ValueError(
                        f"{where} differs from the store array of that name; "
                        "drop it or rename it before write"
                    )
                continue
            new_coords[coord_name] = xr.Variable(
                coord.dims,
                _checked_data(where, coord, strings=True),
                attrs=_checked_attrs(where, coord.attrs),
            )

        attrs = _checked_attrs(f"track {name!r}", value.attrs)
        return xr.Variable(value.dims, data, attrs=attrs), new_coords

    def _drop_reference(self, name: str, value: xr.DataArray) -> xr.DataArray:
        if _coord.NAME not in value.coords:
            return value
        index = value.xindexes.get(_coord.NAME)
        if isinstance(index, _coord.ReferenceIndex):
            same = _coord.same_reference(index.reference, self.reference)
        else:
            coord = value.coords[_coord.NAME]
            same = coord.ndim == 0 and str(coord.values) == _coord.digest(self.reference)
        if not same:
            raise AlignmentError(f"track {name!r} uses another reference than this store")
        return value.drop_vars(_coord.NAME)

    def _labels(self, dim: str) -> np.ndarray:
        if dim in self.data.coords:
            return self.data[dim].values
        group = zarr.open_group(
            self.source, mode="r", use_consolidated=False,
            **_storage_kwargs(self.storage_options),
        )
        return np.asarray(group[dim][:])


def open(
    source,
    *,
    tracks: Sequence[str] | None = None,
    chunks=_STORED_CHUNKS,
    storage_options: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Open a PBZ store as one ``xr.Dataset``.

    Every array carries the 0-d ``reference`` coordinate. ``tracks=`` limits
    the data variables and keeps the reference and label coordinates. By
    default each track is a Dask array chunked by its stored shard (or chunk)
    shape; ``chunks=None`` gives lazy NumPy-backed arrays, and any other value
    goes to ``xr.open_zarr``. The Dataset is a snapshot: a change in memory
    does not reach the store.
    """
    data = _read(source, storage_options, chunks).data
    if tracks is not None:
        try:
            data = _select(data, [tracks] if isinstance(tracks, str) else list(tracks))
        except Exception:
            data.close()
            raise
    return data


def create(
    destination,
    *,
    reference: Mapping[str, int],
    storage_options: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Create an empty PBZ store and return it opened.

    Use it to make a store for ``write``. The import functions do not need
    it: they create the store from the source headers. ``reference`` is the
    ordered mapping of contig name to length that every track shares. The
    destination must be a local path that does not exist. Remote stores are
    not supported yet, so ``storage_options`` must be None.
    """
    path = os.fspath(destination)
    if not path:
        raise ValueError("destination must not be empty")
    if "://" in path or storage_options is not None:
        raise ValueError(
            "create makes local stores only; give a local path and no storage_options"
        )
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
    return open(path, storage_options=storage_options)


def write(
    target,
    variables: Mapping[str, xr.DataArray],
    *,
    encoding=None,
    storage_options: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Add new variables to a store and return the store reopened.

    ``target`` is a path, a URL, or a Dataset from ``open`` (its
    ``encoding["source"]`` names the store). ``variables`` maps names to
    ``xr.DataArray`` values; an ``xr.Dataset`` is such a mapping. The checks
    use the store metadata of now. A genomic dimension (``position`` or
    ``bin_N``) must come first and span the reference. Shared non-genomic
    dimensions must match the store's labels. ``write`` never changes an
    existing variable, and the Dataset passed as ``target`` does not change.
    If the write fails, the arrays it created are removed; it is not atomic
    against a crash.
    """
    source = _target_source(target)
    if not isinstance(variables, Mapping) or not variables:
        raise TypeError("variables must be a nonempty mapping of name to DataArray")
    encoding = dict(encoding or {})

    store = _read(source, storage_options, None)
    try:
        values = {}
        coords = {}
        for name, value in variables.items():
            variable, new_coords = store.check(name, value)
            values[name] = variable
            for coord_name, coord in new_coords.items():
                if coord_name in coords and not coords[coord_name].equals(coord):
                    raise ValueError(
                        f"variables give different values for coordinate {coord_name!r}"
                    )
            coords.update(new_coords)
    finally:
        store.data.close()
    for name in values:
        if name in coords:
            raise ValueError(f"{name!r} is both a variable and a coordinate")
    unknown = set(encoding) - set(values)
    if unknown:
        raise ValueError(f"encoding names variables not being written: {sorted(unknown)}")

    new = xr.Dataset(values, coords=coords, attrs=store.root_attrs)
    var_encoding = {}
    for name in values:
        entry = dict(encoding.get(name, {}))
        entry.setdefault("_FillValue", None)
        var_encoding[name] = entry
    try:
        new.to_zarr(
            store.source,
            mode="a",
            consolidated=False,
            encoding=var_encoding,
            storage_options=_options(storage_options),
        )
    except BaseException:
        store.remove([*values, *coords])
        raise
    return open(store.source, storage_options=storage_options)


def _target_source(target):
    if isinstance(target, xr.Dataset):
        source = target.encoding.get("source")
        if source is None:
            raise ValueError(
                "this Dataset has no encoding['source']; pass the store path or URL"
            )
        return source
    if isinstance(target, xr.DataArray):
        raise TypeError("target must be a path, a URL, or a Dataset from pbzarr.open")
    return target


def _read(source, storage_options, chunks) -> _Store:
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
        data = data.set_coords([name for name in _REFERENCE_ARRAYS if name in data.data_vars])
        reference = _reference(data)
        _check_genomic_dims(data, reference)
        data = _coord.attach(data, reference)
        if chunks is _STORED_CHUNKS:
            data = data.assign(
                {name: _stored_chunks(value) for name, value in data.data_vars.items()}
            )
        for name, value in data.data_vars.items():
            _attach_source(value, source, name, catalog[name], storage_options)
    except Exception:
        data.close()
        raise
    return _Store(
        source,
        data=data,
        root_attrs=root_attrs,
        reference=reference,
        storage_options=storage_options,
        catalog=catalog,
    )


def _checked_data(where: str, value: xr.DataArray, *, strings: bool):
    dtype = np.dtype(value.dtype)
    if dtype.name in _SUPPORTED_DTYPES:
        return value.data
    if strings and dtype.kind in "UT":
        return value.data.astype(object)
    if strings and dtype.kind == "O" and all(
        isinstance(item, str) for item in np.asarray(value.values).ravel().tolist()
    ):
        return value.data
    if strings:
        raise TypeError(f"{where} has unsupported dtype {value.dtype}; use numeric, bool, or str")
    raise TypeError(
        f"{where} has unsupported dtype {value.dtype}; a track is numeric or bool. "
        "Strings are allowed only on variables with no genomic dimension"
    )


def _checked_attrs(where: str, attrs: Mapping) -> dict:
    # pbz:summary and the zarrs provenance note describe the source array,
    # and Xarray keeps attributes through arithmetic.
    attrs = {k: v for k, v in attrs.items() if k not in {"pbz:summary", "_zarrs"}}
    for key in attrs:
        if key.startswith("pbz:") or key in {"zarr_conventions", "coordinates"}:
            raise ValueError(f"{where}: attribute {key!r} is reserved")
        units = attrs[key] if key == "units" else None
        if key in _DECODING_ATTRS or (
            isinstance(units, str) and (" since " in units or units in _TIME_UNITS)
        ):
            raise ValueError(
                f"{where}: attribute {key!r} changes decoding; "
                "pass _FillValue through encoding= instead"
            )
    try:
        json.dumps(attrs)
    except TypeError as error:
        raise TypeError(f"{where}: attributes must be JSON-serializable") from error
    return attrs


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


_LEGACY_MESSAGE = (
    "This is a format 0.4 store. pbzarr 0.8 reads format 0.5 only. "
    "Use pbzarr 0.7, or import the data again."
)


def _check_identity(attrs: Mapping) -> None:
    conventions = attrs.get("zarr_conventions")
    uuids = [
        entry.get("uuid")
        for entry in conventions if isinstance(entry, Mapping)
    ] if isinstance(conventions, list) else []
    matches = uuids.count(_PBZ_UUID)
    if matches == 0 and _LEGACY_UUID in uuids:
        raise PbzError(_LEGACY_MESSAGE)
    if matches != 1:
        raise PbzError("not a valid PBZ store: missing PBZ convention identity")
    version = attrs.get("pbz:version")
    if not isinstance(version, str):
        raise PbzError("not a valid PBZ store: pbz:version is missing or is not a string")
    if version == "0.4":
        raise PbzError(_LEGACY_MESSAGE)
    if version != FORMAT_VERSION:
        raise PbzError(f"Unsupported format version {version}. This pbzarr reads 0.5.")


def _reference(data: xr.Dataset) -> Reference:
    for name in ("contig_id", "contig_length"):
        if name not in data.variables:
            raise PbzError(f"invalid PBZ store: reference array {name!r} is missing")
        if data[name].dims != ("contig",):
            raise PbzError(f"invalid PBZ store: {name!r} must have the one dimension 'contig'")
    return Reference(data["contig_id"].values.tolist(), data["contig_length"].values)


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
