"""Throwaway local Zarr v3 persistence for the Dataset/Track experiment.

Only new variables are supported. Regional writes use fill-backed metadata and
aligned chunk writes, not a materialized full-reference array. No atomicity,
concurrent writers, coordinate append, or selected-column scatter is promised.
"""

from __future__ import annotations

from functools import reduce
from math import gcd
from pathlib import Path
import warnings

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from .genomic_index import Reference, get_index, genomic_width


PBZ_ATTRS = {
    "zarr_conventions": [{"uuid": "d5bd7bea-0388-4e0a-9844-707c034e9add", "name": "pbz"}],
    "pbz:version": "0.5",
    "pbz:reference": {"names": "contig", "lengths": "contig_length"},
}


def _reference(dataset):
    if dataset.attrs.get("zarr_conventions") != PBZ_ATTRS["zarr_conventions"]:
        raise ValueError("Expected the PBZ prototype convention marker.")
    if dataset.attrs.get("pbz:version") != PBZ_ATTRS["pbz:version"]:
        raise ValueError("Expected the isolated format-0.5 prototype format.")
    if dataset.attrs.get("pbz:reference") != PBZ_ATTRS["pbz:reference"]:
        raise ValueError("This prototype requires contig/contig_length reference arrays.")
    if not {"contig", "contig_length"}.issubset(dataset.coords):
        raise ValueError("Reference arrays must decode as coordinates.")
    if any(dataset[name].dims != ("contig",) for name in ("contig", "contig_length")):
        raise ValueError("Reference arrays must use the contig dimension.")
    return Reference(tuple(dataset.contig.values), tuple(dataset.contig_length.values))


def _check_labels(coord, dim):
    if coord.dims != (dim,):
        raise ValueError(f"{dim!r} needs its own one-dimensional label coordinate.")
    index = coord.to_index()
    if not index.is_unique or index.hasnans:
        raise ValueError(f"{dim!r} labels must be unique and nonmissing.")


def create_store(path, reference, coords=None, attrs=None):
    """Create an empty scratch Dataset; an existing path is always an error."""
    if Path(path).exists():
        raise FileExistsError(path)
    if not isinstance(reference, Reference):
        reference = Reference.from_mapping(reference)
    attrs = dict(attrs or {})
    if (set(PBZ_ATTRS) | {"coordinates"}).intersection(attrs):
        raise ValueError("User attributes cannot replace convention fields.")
    coords = dict(coords or {})
    if {"contig", "contig_length"}.intersection(coords):
        raise ValueError("Reference coordinates come from reference=.")
    dataset = xr.Dataset(coords={
        "contig": ("contig", np.asarray(reference.names, dtype=object)),
        "contig_length": ("contig", np.asarray(reference.lengths, dtype="int64")),
        **coords,
    }, attrs={**attrs, **PBZ_ATTRS})
    for dim in dataset.dims:
        if genomic_width(dim) is not None:
            raise ValueError("Genomic coordinates are implicit and cannot be supplied.")
        if dim not in dataset.coords:
            raise ValueError(f"Missing labels for {dim!r}.")
        _check_labels(dataset[dim], dim)
    for name, coord in list(dataset.coords.items()):
        if coord.dtype.kind == "U":
            dataset = dataset.assign_coords({name: coord.astype(object)})
    dataset.to_zarr(path, mode="w-", zarr_format=3, consolidated=False)


def read_store(path, chunks="auto"):
    """Return a native Dataset without attaching runtime genomic indexes."""
    dataset = xr.open_zarr(path, chunks=chunks, consolidated=False, zarr_format=3)
    try:
        _reference(dataset)
    except Exception:
        dataset.close()
        raise
    return dataset


def _regional_chunks(array, index, encoding):
    requested = encoding.get("chunks")
    if requested is None:
        chunks = []
        for axis, size in enumerate(array.shape):
            boundaries = list(array.chunks[axis]) if array.chunks else [size]
            if axis == 0:
                boundaries.extend([index.rows.start, len(index.rows)])
            chunks.append(max(1, reduce(gcd, boundaries)))
        requested = tuple(chunks)
    chunks = tuple(requested)
    if len(chunks) != array.ndim or any(not isinstance(c, (int, np.integer)) or c <= 0 for c in chunks):
        raise ValueError("Storage chunks must be a positive integer per dimension.")
    start, stop = index.rows.start, index.rows.stop
    extent = index.reference.size(index.width)
    if start % chunks[0] or (stop != extent and stop % chunks[0]):
        raise ValueError("Regional writes require chunk-aligned start and end boundaries.")
    return chunks


def write_tracks(path, current, tracks, encoding=None):
    """Add a batch of full or contiguous regional tracks to an existing store."""
    encoding = dict(encoding or {})
    if set(encoding) - set(tracks):
        raise ValueError("Encoding names must identify tracks in this write.")
    # Consult the complete on-disk catalog, including tracks a caller filtered out.
    with read_store(path, chunks=None) as stored:
        reference = _reference(stored)
        if _reference(current) != reference:
            raise ValueError("The writable handle has a different reference.")
        labels = {dim: stored[dim] for dim in stored.dims if genomic_width(dim) is None}
        existing_names = set(stored.variables) | set(stored.dims)
        existing_coords = set(stored.coords)
        coordinates = dict(stored.coords)
        raw_attrs = dict(zarr.open_group(path, mode="r").attrs)
        plans = []
        if set(tracks).intersection({coord for array in tracks.values() for coord in array.coords}):
            raise ValueError("A planned track name collides with a coordinate.")
        for name, array in tracks.items():
            if not isinstance(name, str) or not name or "/" in name or name in existing_names:
                raise ValueError(f"Invalid or existing destination name: {name!r}.")
            if name in array.dims or name in array.coords:
                raise ValueError("A track name cannot also be a dimension or coordinate.")
            index = get_index(array)
            if index is not None:
                if index.reference != reference:
                    raise ValueError(f"{name!r} has a different reference.")
                if index.rows.step != 1:
                    raise ValueError("This writer supports only contiguous increasing genomic rows.")
                array = array.transpose(index.dim, *(d for d in array.dims if d != index.dim))
                array = array.drop_vars(index.dim)
            for dim in array.dims:
                if genomic_width(dim) is not None:
                    continue
                if dim not in array.coords:
                    raise ValueError(f"{name!r} has no labels for {dim!r}.")
                _check_labels(array[dim], dim)
                if dim in labels and not array[dim].equals(labels[dim]):
                    raise ValueError(f"{dim!r} labels must match exactly; column scatter is not supported.")
                labels[dim] = array[dim]
            for coord in set(array.coords).intersection(coordinates):
                if not array[coord].equals(coordinates[coord]):
                    raise ValueError(f"Existing coordinate {coord!r} differs.")
            if set(array.coords).intersection(existing_names - existing_coords - set(labels)):
                raise ValueError("A new coordinate collides with a stored or planned track.")
            # Do not inherit source storage layout or stale coordinate encoding.
            clean = xr.DataArray(array.data, dims=array.dims, coords=array.coords, attrs=dict(array.attrs))
            clean.encoding.clear()
            options = dict(encoding.get(name, {}))
            regional = index is not None and index.rows != range(reference.size(index.width))
            if regional:
                if not len(index.rows):
                    raise ValueError("Empty regional writes are outside this prototype.")
                if "fill_value" not in options:
                    raise ValueError("Regional new-track writes require an explicit fill_value.")
                if any(index.dim in coord.dims for coord in clean.coords.values()):
                    raise ValueError("Regional genomic auxiliary coordinates are outside this prototype.")
                options["chunks"] = _regional_chunks(clean, index, options)
            plans.append((name, clean, index, regional, options))
            existing_names.add(name)
            coordinates.update(array.coords)

        if set(tracks).intersection(labels):
            raise ValueError("A planned track name collides with a dimension coordinate.")

        # No measurement computation or store mutation occurred during preflight.
        for name, array, index, regional, options in plans:
            payload = array.to_dataset(name=name)
            payload = payload.drop_vars(set(payload.coords).intersection(existing_coords))
            payload.attrs = raw_attrs.copy()  # Preserve root coordinate classification.
            if not regional:
                payload.to_zarr(path, mode="a", encoding={name: options}, zarr_format=3,
                                consolidated=False, safe_chunks=True)
            else:
                shape = (reference.size(index.width), *array.shape[1:])
                placeholder = da.full(shape, options["fill_value"], dtype=array.dtype, chunks=options["chunks"])
                template = xr.Dataset({name: (array.dims, placeholder, array.attrs)}, attrs=raw_attrs.copy())
                for coord in payload.coords:
                    template = template.assign_coords({coord: payload[coord]})
                template.to_zarr(path, mode="a", encoding={name: options}, compute=False,
                                 zarr_format=3, consolidated=False, safe_chunks=True)
                values = xr.Dataset({name: (array.dims, array.data, array.attrs)})
                values = values.chunk(dict(zip(array.dims, options["chunks"])))
                values.to_zarr(path, mode="r+", region={index.dim: slice(index.rows.start, index.rows.stop)},
                               zarr_format=3, consolidated=False, safe_chunks=True)
                warnings.warn(f"{name!r} uses fill {options['fill_value']!r} outside its selected genomic rows.",
                              UserWarning, stacklevel=2)
            existing_coords.update(payload.coords)
