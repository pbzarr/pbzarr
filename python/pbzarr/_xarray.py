"""PBZ validation for datasets opened by xarray's Zarr backend."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from ._native import PbzError


_NATIVE_CHUNKS = object()

_RESERVED_NAMES = frozenset(
    {
        "position",
        "contig",
        "contig_boundary",
        "contigs",
        "offsets",
        "region",
        "region_boundary",
        "region_contig",
        "region_start",
        "region_stop",
        "region_input_index",
        "region_storage_index",
        "values",
    }
)
_CHECKSUM_PATTERN = re.compile(r"md5:[0-9a-f]{32}\Z")


def _open_zarr(source, *, group=None, storage_options=None) -> xr.Dataset:
    kwargs = {}
    if storage_options is not None:
        kwargs["storage_options"] = dict(storage_options)
    return xr.open_zarr(
        source,
        group=group,
        chunks=None,
        consolidated=False,
        create_default_indexes=False,
        decode_cf=False,
        **kwargs,
    )


def _node_kind(attrs: Mapping) -> str:
    conventions = attrs.get("zarr_conventions", ())
    if not isinstance(conventions, list) or not any(
        isinstance(convention, Mapping) and convention.get("name") == "perbase"
        for convention in conventions
    ):
        raise PbzError("invalid PBZ node: missing perbase convention marker")

    if "perbase:kind" in attrs:
        kind = attrs["perbase:kind"]
        if not isinstance(kind, str) or kind not in {"track", "collection"}:
            raise PbzError(
                "invalid PBZ node: perbase:kind must be 'track' or 'collection'"
            )
        return kind
    return "track" if "perbase:version" in attrs else "collection"


def _prepare_track_dataset(opened: xr.Dataset, *, chunks) -> xr.Dataset:
    try:
        dataset = _classify_track_coordinates(opened)
        _validate_track(dataset)
        dataset = dataset.assign_attrs({**dataset.attrs, "perbase:kind": "track"})
        source_values = dataset["values"]
        values = xr.DataArray(source_values.variable, name="values")
        values.encoding = source_values.encoding
        if chunks is _NATIVE_CHUNKS or isinstance(chunks, Mapping) and not chunks:
            values = _apply_default_chunks(values)
        elif chunks is not None:
            values = values.chunk(chunks)
        dataset = dataset.assign(values=values.variable)
        dataset.encoding.pop("source", None)
        dataset.set_close(opened.close)
        return dataset
    except Exception:
        opened.close()
        raise


def _classify_track_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    coordinate_names = [
        name for name in ("contigs", "offsets") if name in dataset.variables
    ]
    values = dataset.variables.get("values")
    if values is not None and values.ndim == 2:
        column_dim = values.dims[1]
        if column_dim in dataset.variables:
            coordinate_names.append(column_dim)

    for name in coordinate_names:
        dataset[name].load()
    classified = dataset.set_coords(coordinate_names)
    classified.set_close(dataset.close)
    return classified


def _validate_track(dataset: xr.Dataset) -> None:
    attrs = dataset.attrs
    if _node_kind(attrs) != "track":
        raise PbzError("PBZ node is a collection, not a track")
    if attrs.get("perbase:version") != "0.4":
        raise PbzError("invalid PBZ track: unsupported perbase:version")
    if attrs.get("perbase:coordinates") != "0-based-half-open":
        raise PbzError("invalid PBZ track: unsupported coordinate convention")

    for name in ("values", "contigs", "offsets"):
        if name not in dataset.variables:
            raise PbzError(f"invalid PBZ track: missing {name!r} array")

    values = dataset["values"]
    if values.ndim not in {1, 2}:
        raise PbzError("invalid PBZ track: values must have rank one or two")
    if values.dims[0] != "position":
        raise PbzError("invalid PBZ track: position must be the first values dimension")
    if dataset["contigs"].dims != ("contig",):
        raise PbzError("invalid PBZ track: contigs must use the contig dimension")
    if dataset["offsets"].dims != ("contig_boundary",):
        raise PbzError(
            "invalid PBZ track: offsets must use the contig_boundary dimension"
        )

    allowed_arrays = {"values", "contigs", "offsets"}
    if values.ndim == 2:
        column_dim = values.dims[1]
        if column_dim in _RESERVED_NAMES:
            raise PbzError(
                f"invalid PBZ track: reserved column dimension {column_dim!r}"
            )
        if column_dim not in dataset.variables:
            raise PbzError(
                f"invalid PBZ track: missing {column_dim!r} column labels"
            )
        labels = dataset[column_dim]
        if labels.dims != (column_dim,) or labels.size != values.shape[1]:
            raise PbzError(
                "invalid PBZ track: column labels must match the values column axis"
            )
        allowed_arrays.add(column_dim)

    unexpected_arrays = sorted(set(dataset.variables) - allowed_arrays)
    if unexpected_arrays:
        raise PbzError(
            f"invalid PBZ track: unexpected array {unexpected_arrays[0]!r}"
        )

    raw_contigs = np.asarray(dataset["contigs"].values)
    offsets = np.asarray(dataset["offsets"].values)
    if not np.issubdtype(offsets.dtype, np.integer) or np.issubdtype(
        offsets.dtype, np.bool_
    ):
        raise PbzError("invalid PBZ track: offsets must be integers")
    if offsets.ndim != 1 or offsets.size != raw_contigs.size + 1:
        raise PbzError(
            "invalid PBZ track: offsets must have one more entry than contigs"
        )

    contigs = raw_contigs.tolist()
    if any(not isinstance(contig, str) or not contig for contig in contigs):
        raise PbzError("invalid PBZ track: contig names must be nonempty strings")
    if len(set(contigs)) != len(contigs):
        raise PbzError("invalid PBZ track: contig names must be unique")
    if offsets[0] != 0:
        raise PbzError("invalid PBZ track: offsets must start at zero")
    if np.any(offsets[1:] < offsets[:-1]):
        raise PbzError("invalid PBZ track: offsets must be nondecreasing")
    if int(offsets[-1]) != values.shape[0]:
        raise PbzError(
            "invalid PBZ track: terminal offset must equal the values length"
        )

    if values.ndim == 2:
        labels = np.asarray(dataset[values.dims[1]].values).tolist()
        if any(not isinstance(label, str) for label in labels):
            raise PbzError("invalid PBZ track: column labels must be strings")
        if len(set(labels)) != len(labels):
            raise PbzError("invalid PBZ track: column labels must be unique")

    checksum = attrs.get("perbase:genome_checksum")
    if not isinstance(checksum, str) or _CHECKSUM_PATTERN.fullmatch(checksum) is None:
        raise PbzError("invalid PBZ track: malformed perbase:genome_checksum")
    records = sorted(
        zip(contigs, np.diff(offsets), strict=True),
        key=lambda item: item[0].encode("utf-8"),
    )
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    computed = "md5:" + hashlib.md5(payload.encode("utf-8")).hexdigest()
    if checksum != computed:
        raise PbzError("invalid PBZ track: genome checksum does not match geometry")


def _apply_default_chunks(values: xr.DataArray) -> xr.DataArray:
    widths = values.encoding.get("shards")
    if widths is None:
        widths = values.encoding.get("chunks")
    if widths is None:
        return values.chunk()
    return values.chunk(dict(zip(values.dims, widths, strict=True)))


def _normalize_track_names(
    tracks: Sequence[str], available: Mapping[str, object] | None = None
) -> list[str]:
    if isinstance(tracks, (str, bytes)):
        raise PbzError("tracks must be a sequence of direct child names")
    try:
        names = list(tracks)
    except TypeError as error:
        raise PbzError("tracks must be a sequence of direct child names") from error

    seen = set()
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
        ):
            raise PbzError("track names must identify direct collection children")
        if name in seen:
            raise PbzError(f"duplicate track name {name!r}")
        if available is not None and name not in available:
            raise PbzError(f"unknown track {name!r}")
        seen.add(name)
    return names


def _compose_track_datasets(datasets: Mapping[str, xr.Dataset]) -> xr.Dataset:
    items = list(datasets.items())
    first_name, first = items[0]
    checksum = first.attrs["perbase:genome_checksum"]
    contigs = np.asarray(first["contigs"].values)
    offsets = np.asarray(first["offsets"].values)

    namespace = set()
    column_coordinates = {}
    data_variables = {}
    genome_names = []
    for name, dataset in items:
        if dataset.attrs["perbase:genome_checksum"] != checksum:
            raise PbzError(
                f"track {name!r} has a different genome checksum than {first_name!r}"
            )
        if not np.array_equal(dataset["contigs"].values, contigs) or not np.array_equal(
            dataset["offsets"].values, offsets
        ):
            raise PbzError(
                f"track {name!r} has different ordered genome geometry than "
                f"{first_name!r}"
            )

        namespace.update(dataset.dims)
        namespace.update(dataset.coords)
        values = dataset.variables["values"]
        data_variables[name] = values
        if values.ndim == 2:
            column_dim = values.dims[1]
            labels = dataset.variables[column_dim]
            previous = column_coordinates.get(column_dim)
            if previous is not None and not np.array_equal(
                previous.values, labels.values
            ):
                raise PbzError(
                    f"tracks using {column_dim!r} have different column labels"
                )
            column_coordinates[column_dim] = labels
        genome_names.append(dataset.attrs.get("perbase:genome_name"))

    collisions = sorted(set(datasets) & namespace)
    if collisions:
        raise PbzError(
            f"track name {collisions[0]!r} collides with a result dimension or coordinate"
        )

    coordinates = xr.Coordinates(
        {
            "contigs": first.variables["contigs"],
            "offsets": first.variables["offsets"],
            **column_coordinates,
        },
        indexes={},
    )
    attrs = {
        "genome_checksum": checksum,
        "coordinates": "0-based-half-open",
    }
    if genome_names[0] is not None and all(
        name == genome_names[0] for name in genome_names
    ):
        attrs["genome_name"] = genome_names[0]

    return xr.Dataset(data_vars=data_variables, coords=coordinates, attrs=attrs)
