"""Normalization and geometry resolution for PBZ region operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import dask.array as da
from dask import delayed
import numpy as np
import xarray as xr

from ._native import PbzError
from ._region import RegionQuery, parse_region
from ._xarray import _validate_track


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_I32_MAX = 2**31 - 1
_TARGET_BYTES = 128 * 1024**2
_MAX_SOURCE_BLOCKS = 16


@dataclass(frozen=True)
class RegionLayout:
    contig_ids: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    flat_starts: np.ndarray
    flat_stops: np.ndarray
    packed_offsets: np.ndarray
    input_index: np.ndarray

    @property
    def n_regions(self) -> int:
        return self.contig_ids.size

    @property
    def total_positions(self) -> int:
        return int(self.packed_offsets[-1])


def _plan_variable_regions(
    values: xr.DataArray,
    layout: RegionLayout,
    *,
    target_bytes: int,
    max_source_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return batch_region_edges, batch_piece_edges, and structured pieces."""
    if values.ndim not in {1, 2} or values.dims[0] != "position":
        raise PbzError("packed region values must have position first")
    if any("position" in coordinate.dims for coordinate in values.coords.values()):
        raise PbzError(
            "packed region values cannot have position-dependent coordinates"
        )
    if (
        isinstance(target_bytes, (bool, np.bool_))
        or not isinstance(target_bytes, (int, np.integer))
        or target_bytes <= 0
    ):
        raise ValueError("target_bytes must be a positive integer")
    if (
        isinstance(max_source_blocks, (bool, np.bool_))
        or not isinstance(max_source_blocks, (int, np.integer))
        or max_source_blocks <= 0
    ):
        raise ValueError("max_source_blocks must be a positive integer")

    chunks = values.chunks
    if chunks is None:
        position_chunks = (values.shape[0],)
    else:
        position_chunks = tuple(int(width) for width in chunks[0])
    boundaries = np.empty(len(position_chunks) + 1, dtype=np.int64)
    boundaries[0] = 0
    np.cumsum(position_chunks, out=boundaries[1:])
    if int(boundaries[-1]) != values.shape[0]:
        raise PbzError("position chunks must cover the values position axis")
    if (
        np.any(layout.flat_starts < 0)
        or np.any(layout.flat_stops <= layout.flat_starts)
        or np.any(layout.flat_stops > values.shape[0])
    ):
        raise PbzError("region layout is outside the values position axis")

    first_blocks = np.searchsorted(
        boundaries[1:], layout.flat_starts, side="right"
    )
    last_blocks = np.searchsorted(
        boundaries[1:], layout.flat_stops - 1, side="right"
    )
    piece_counts = last_blocks - first_blocks + 1
    region_piece_edges = np.empty(layout.n_regions + 1, dtype=np.int64)
    region_piece_edges[0] = 0
    np.cumsum(piece_counts, out=region_piece_edges[1:])
    n_pieces = int(region_piece_edges[-1])

    piece_regions = np.repeat(
        np.arange(layout.n_regions, dtype=np.int64), piece_counts
    )
    within_regions = np.arange(n_pieces, dtype=np.int64) - np.repeat(
        region_piece_edges[:-1], piece_counts
    )
    piece_blocks = np.repeat(first_blocks, piece_counts) + within_regions
    piece_flat_starts = np.maximum(
        layout.flat_starts[piece_regions], boundaries[piece_blocks]
    )
    piece_flat_stops = np.minimum(
        layout.flat_stops[piece_regions], boundaries[piece_blocks + 1]
    )
    source_starts = piece_flat_starts - boundaries[piece_blocks]
    source_stops = piece_flat_stops - boundaries[piece_blocks]

    bytes_per_position = int(np.dtype(values.dtype).itemsize)
    if values.ndim == 2:
        bytes_per_position *= values.shape[1]
    region_edges = np.empty(layout.n_regions + 1, dtype=np.int64)
    region_edges[0] = 0
    edge_count = 1
    batch_bytes = 0
    batch_pieces = 0
    batch_start = 0
    for region_index in range(layout.n_regions):
        next_bytes = (
            int(
                layout.packed_offsets[region_index + 1]
                - layout.packed_offsets[region_index]
            )
            * bytes_per_position
        )
        next_pieces = int(piece_counts[region_index])
        if region_index > batch_start and (
            batch_bytes + next_bytes > target_bytes
            or batch_pieces + next_pieces > max_source_blocks
        ):
            region_edges[edge_count] = region_index
            edge_count += 1
            batch_start = region_index
            batch_bytes = 0
            batch_pieces = 0
        batch_bytes += next_bytes
        batch_pieces += next_pieces
    region_edges[edge_count] = layout.n_regions
    edge_count += 1
    region_edges = region_edges[:edge_count]
    piece_edges = region_piece_edges[region_edges]

    largest_bound = max(
        layout.n_regions,
        n_pieces,
        len(position_chunks),
        int(boundaries[-1]),
    )
    plan_dtype = np.dtype(np.int32 if largest_bound <= _I32_MAX else np.int64)
    batch_region_edges = region_edges.astype(plan_dtype, copy=False)
    batch_piece_edges = piece_edges.astype(plan_dtype, copy=False)
    piece_dtype = np.dtype(
        [
            ("source_block", plan_dtype),
            ("source_start", plan_dtype),
            ("source_stop", plan_dtype),
        ]
    )
    pieces = np.empty(n_pieces, dtype=piece_dtype)
    pieces["source_block"] = piece_blocks
    pieces["source_start"] = source_starts
    pieces["source_stop"] = source_stops

    piece_lengths = source_stops - source_starts
    batch_positions = (
        layout.packed_offsets[region_edges[1:]]
        - layout.packed_offsets[region_edges[:-1]]
    )
    max_batch_positions = (
        target_bytes // bytes_per_position if bytes_per_position else _I64_MAX
    )
    within_limits = (batch_positions <= max_batch_positions) & (
        np.diff(piece_edges) <= max_source_blocks
    )
    valid = (
        batch_region_edges[0] == 0
        and batch_region_edges[-1] == layout.n_regions
        and batch_piece_edges[0] == 0
        and batch_piece_edges[-1] == n_pieces
        and np.all(np.diff(batch_region_edges) > 0)
        and np.all(np.diff(batch_piece_edges) > 0)
        and np.array_equal(piece_edges, region_piece_edges[region_edges])
        and np.all(piece_blocks >= 0)
        and np.all(piece_blocks < len(position_chunks))
        and np.all(source_starts >= 0)
        and np.all(source_stops > source_starts)
        and np.all(
            source_stops
            <= boundaries[piece_blocks + 1] - boundaries[piece_blocks]
        )
        and np.array_equal(
            np.add.reduceat(piece_lengths, region_piece_edges[:-1]),
            np.diff(layout.packed_offsets),
        )
        and int(piece_lengths.sum()) == layout.total_positions
        and np.all(within_limits | (np.diff(region_edges) == 1))
    )
    if not valid:
        raise AssertionError("invalid packed region plan")

    for array in (batch_region_edges, batch_piece_edges, pieces):
        array.setflags(write=False)
    return batch_region_edges, batch_piece_edges, pieces


def _gather_block(source_blocks, pieces, output_shape) -> np.ndarray:
    if len(source_blocks) != len(pieces):
        raise ValueError("source blocks and pieces must have equal lengths")
    if not source_blocks:
        raise ValueError("a gather block must contain at least one piece")

    first = np.asarray(source_blocks[0])
    output = np.empty(output_shape, dtype=first.dtype)
    destination = 0
    for source, piece in zip(source_blocks, pieces, strict=True):
        start = int(piece["source_start"])
        stop = int(piece["source_stop"])
        length = stop - start
        output[destination : destination + length] = np.asarray(source)[start:stop]
        destination += length
    if destination != output_shape[0]:
        raise ValueError("piece lengths do not cover the gather output")
    return output


def regions_dataset(
    ds: xr.Dataset,
    intervals,
    *,
    target_bytes: int = _TARGET_BYTES,
    max_source_blocks: int = _MAX_SOURCE_BLOCKS,
) -> xr.Dataset:
    layout = _resolve_regions(ds, intervals)

    data_variables = {}
    column_coordinates = {}
    for name, values in ds.data_vars.items():
        batch_regions, batch_pieces, pieces = _plan_variable_regions(
            values,
            layout,
            target_bytes=target_bytes,
            max_source_blocks=max_source_blocks,
        )
        if values.chunks is None:
            gathered = _gather_eager(values, pieces, layout.total_positions)
        else:
            gathered = _gather_dask(
                values,
                layout,
                batch_regions,
                batch_pieces,
                pieces,
            )
        data_variables[name] = xr.Variable(
            values.dims,
            gathered,
            attrs=values.attrs,
        )
        for dimension in values.dims[1:]:
            column_coordinates[dimension] = ds[dimension].variable

    contigs = np.asarray(ds["contigs"].values)
    coordinates = xr.Coordinates(
        {
            "offsets": ("region_boundary", layout.packed_offsets),
            "region_contig": ("region", contigs[layout.contig_ids]),
            "region_start": ("region", layout.starts),
            "region_stop": ("region", layout.stops),
            "region_input_index": ("region", layout.input_index),
            "region_storage_index": (
                "region",
                np.arange(layout.n_regions, dtype=np.int64),
            ),
            **column_coordinates,
        },
        indexes={},
    )
    checksum = ds.attrs.get(
        "perbase:genome_checksum", ds.attrs.get("genome_checksum")
    )
    result = xr.Dataset(
        data_vars=data_variables,
        coords=coordinates,
        attrs={
            "pbz:representation": "packed-regions",
            "pbz:parent_genome_checksum": checksum,
            "pbz:coordinates": "0-based-half-open",
        },
    )
    result.set_close(ds.close)
    return result


def _gather_dask(
    values: xr.DataArray,
    layout: RegionLayout,
    batch_region_edges: np.ndarray,
    batch_piece_edges: np.ndarray,
    pieces: np.ndarray,
):
    source_blocks = values.data.to_delayed()
    rows = []
    for batch_index in range(len(batch_region_edges) - 1):
        region_start = int(batch_region_edges[batch_index])
        region_stop = int(batch_region_edges[batch_index + 1])
        piece_start = int(batch_piece_edges[batch_index])
        piece_stop = int(batch_piece_edges[batch_index + 1])
        batch_view = pieces[piece_start:piece_stop]
        position_size = int(
            layout.packed_offsets[region_stop]
            - layout.packed_offsets[region_start]
        )
        if values.ndim == 1:
            selected = tuple(
                source_blocks[(int(piece["source_block"]),)]
                for piece in batch_view
            )
            task = delayed(_gather_block)(
                selected,
                batch_view,
                (position_size,),
            )
            rows.append(
                da.from_delayed(
                    task,
                    shape=(position_size,),
                    dtype=values.dtype,
                )
            )
            continue

        columns = []
        for column_block, column_size in enumerate(values.chunks[1]):
            selected = tuple(
                source_blocks[
                    (int(piece["source_block"]), column_block)
                ]
                for piece in batch_view
            )
            task = delayed(_gather_block)(
                selected,
                batch_view,
                (position_size, int(column_size)),
            )
            columns.append(
                da.from_delayed(
                    task,
                    shape=(position_size, int(column_size)),
                    dtype=values.dtype,
                )
            )
        rows.append(
            columns[0]
            if len(columns) == 1
            else da.concatenate(columns, axis=1)
        )
    return rows[0] if len(rows) == 1 else da.concatenate(rows, axis=0)


def _gather_eager(
    values: xr.DataArray,
    pieces: np.ndarray,
    total_positions: int,
) -> np.ndarray:
    selected = []
    local_pieces = pieces.copy()
    lengths = pieces["source_stop"] - pieces["source_start"]
    for piece in pieces:
        source_start = int(piece["source_start"])
        source_stop = int(piece["source_stop"])
        selected.append(
            np.asarray(
                values.isel(position=slice(source_start, source_stop)).values
            )
        )
    local_pieces["source_start"] = 0
    local_pieces["source_stop"] = lengths
    return _gather_block(
        tuple(selected),
        local_pieces,
        (total_positions, *values.shape[1:]),
    )


def _normalize_one(query) -> RegionQuery:
    if isinstance(query, RegionQuery):
        normalized = query
    elif isinstance(query, str):
        normalized = parse_region(query)
    elif isinstance(query, tuple) and len(query) == 3:
        normalized = RegionQuery(query[0], query[1], query[2])
    else:
        raise TypeError(
            "region query must be a RegionQuery, string, or "
            "(contig, start, stop) tuple"
        )

    if not isinstance(normalized.contig, str) or not normalized.contig:
        raise ValueError("region contig must be a nonempty string")
    return RegionQuery(
        normalized.contig,
        _normalize_coordinate(normalized.start, "start"),
        _normalize_coordinate(normalized.stop, "stop"),
    )


def _normalize_coordinate(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"region {name} must be an integer or None")
    coordinate = int(value)
    if coordinate < _I64_MIN or coordinate > _I64_MAX:
        raise ValueError(f"region {name} is outside signed 64-bit range")
    return coordinate


def _resolve_region(
    ds: xr.Dataset, query
) -> tuple[RegionQuery, slice]:
    if "pbz:representation" in ds.attrs:
        raise PbzError("region() requires a normal PBZ track Dataset")
    _validate_track(ds)
    normalized = _normalize_one(query)
    return _resolve_one(ds, normalized)


def _resolve_one(
    ds: xr.Dataset, normalized: RegionQuery
) -> tuple[RegionQuery, slice]:
    contigs = np.asarray(ds["contigs"].values).tolist()
    try:
        contig_id = contigs.index(normalized.contig)
    except ValueError as error:
        raise KeyError(f"unknown contig {normalized.contig!r}") from error

    offsets = np.asarray(ds["offsets"].values)
    flat_start = int(offsets[contig_id])
    contig_length = int(offsets[contig_id + 1]) - flat_start
    start = 0 if normalized.start is None else normalized.start
    stop = contig_length if normalized.stop is None else normalized.stop

    if start < 0:
        raise ValueError("region start must be nonnegative")
    if stop <= start:
        raise ValueError("region must be nonempty with start less than stop")
    if stop > contig_length:
        raise ValueError(
            f"region stop {stop} exceeds contig length {contig_length}"
        )

    resolved = RegionQuery(normalized.contig, start, stop)
    return resolved, slice(flat_start + start, flat_start + stop)


def _resolve_regions(ds: xr.Dataset, intervals) -> RegionLayout:
    if "pbz:representation" in ds.attrs:
        raise PbzError("regions() requires a normal PBZ track Dataset")
    _validate_regions_dataset(ds)

    if isinstance(intervals, (RegionQuery, str)) or (
        isinstance(intervals, tuple)
        and len(intervals) == 3
        and isinstance(intervals[0], str)
    ):
        normalized = _normalize_one(intervals)
        resolved, flat_slice = _resolve_one(ds, normalized)
        contigs = np.asarray(ds["contigs"].values)
        contig_id = np.flatnonzero(contigs == resolved.contig)
        return _build_region_layout(
            np.asarray([int(contig_id[0])], dtype=np.int64),
            np.asarray([resolved.start], dtype=np.int64),
            np.asarray([resolved.stop], dtype=np.int64),
            np.asarray([flat_slice.start], dtype=np.int64),
            np.asarray([flat_slice.stop], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        )

    columns = _dataframe_columns(intervals)
    if columns is None:
        columns = _array_columns(intervals)
    if columns is None:
        columns = _row_columns(intervals)
    contig_names, starts, stops = _validate_columns(*columns)

    contigs = np.asarray(ds["contigs"].values)
    contig_ids = _contig_ids(contigs, contig_names)
    offsets = np.asarray(ds["offsets"].values, dtype=np.int64)
    contig_lengths = offsets[contig_ids + 1] - offsets[contig_ids]
    if np.any(starts < 0):
        raise ValueError("region start must be nonnegative")
    if np.any(stops <= starts):
        raise ValueError("region must be nonempty with start less than stop")
    if np.any(stops > contig_lengths):
        raise ValueError("region stop exceeds contig length")

    input_index = np.arange(contig_ids.size, dtype=np.int64)
    order = np.lexsort((input_index, starts, contig_ids))
    contig_ids = contig_ids[order]
    starts = starts[order]
    stops = stops[order]
    input_index = input_index[order]
    if np.any((contig_ids[1:] == contig_ids[:-1]) & (starts[1:] < stops[:-1])):
        raise ValueError("regions must not overlap")

    return _build_region_layout(
        contig_ids,
        starts,
        stops,
        offsets[contig_ids] + starts,
        offsets[contig_ids] + stops,
        input_index,
    )


def _validate_regions_dataset(ds: xr.Dataset) -> None:
    if "pbz:representation" in ds.attrs:
        raise PbzError("regions() requires an unpacked PBZ Dataset")
    if "perbase:version" in ds.attrs or ds.attrs.get("perbase:kind") == "track":
        _validate_track(ds)
        return

    allowed_attrs = {"genome_checksum", "coordinates", "genome_name"}
    if set(ds.attrs) - allowed_attrs:
        raise PbzError("regions() requires an exactly composed PBZ Dataset")
    if ds.attrs.get("coordinates") != "0-based-half-open":
        raise PbzError("invalid composed Dataset coordinate convention")
    if not ds.data_vars:
        raise PbzError("composed PBZ Dataset must contain at least one track")
    if "contigs" not in ds.coords or "offsets" not in ds.coords:
        raise PbzError("composed PBZ Dataset is missing genome coordinates")
    if ds["contigs"].dims != ("contig",):
        raise PbzError("composed contigs must use the contig dimension")
    if ds["offsets"].dims != ("contig_boundary",):
        raise PbzError("composed offsets must use the contig_boundary dimension")

    contigs_array = np.asarray(ds["contigs"].values)
    offsets = np.asarray(ds["offsets"].values)
    if not np.issubdtype(offsets.dtype, np.integer) or np.issubdtype(
        offsets.dtype, np.bool_
    ):
        raise PbzError("composed offsets must be integers")
    if offsets.ndim != 1 or offsets.size != contigs_array.size + 1:
        raise PbzError("composed offsets must bound every contig")
    contigs = contigs_array.tolist()
    if any(not isinstance(contig, str) or not contig for contig in contigs):
        raise PbzError("composed contig names must be nonempty strings")
    if len(set(contigs)) != len(contigs):
        raise PbzError("composed contig names must be unique")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise PbzError("composed offsets must be nondecreasing from zero")

    expected_variables = set(ds.data_vars) | {"contigs", "offsets"}
    for name, values in ds.data_vars.items():
        if values.ndim not in {1, 2} or values.dims[0] != "position":
            raise PbzError(f"composed track {name!r} must have position first")
        if values.shape[0] != int(offsets[-1]):
            raise PbzError(f"composed track {name!r} has incompatible geometry")
        if values.ndim == 2:
            column_dim = values.dims[1]
            if column_dim not in ds.coords:
                raise PbzError(
                    f"composed track {name!r} is missing column labels"
                )
            labels = ds[column_dim]
            if labels.dims != (column_dim,) or labels.size != values.shape[1]:
                raise PbzError(
                    f"composed track {name!r} has invalid column labels"
                )
            expected_variables.add(column_dim)
    if set(ds.variables) != expected_variables:
        raise PbzError("regions() requires an exactly composed PBZ Dataset")
    if any("position" in coordinate.dims for coordinate in ds.coords.values()):
        raise PbzError("composed PBZ Dataset has a position-dependent coordinate")

    records = sorted(
        zip(contigs, np.diff(offsets), strict=True),
        key=lambda item: item[0].encode("utf-8"),
    )
    payload = "".join(f"{name}\t{int(length)}\n" for name, length in records)
    checksum = "md5:" + hashlib.md5(payload.encode("utf-8")).hexdigest()
    if ds.attrs.get("genome_checksum") != checksum:
        raise PbzError("composed genome checksum does not match geometry")


def _dataframe_columns(intervals):
    if not (hasattr(intervals, "columns") and hasattr(intervals, "__getitem__")):
        return None
    names = list(intervals.columns)
    contig_names = [name for name in names if name in {"contig", "chrom", "#chrom"}]
    start_names = [name for name in names if name == "start"]
    stop_names = [name for name in names if name in {"stop", "end"}]
    if len(contig_names) != 1 or len(start_names) != 1 or len(stop_names) != 1:
        raise ValueError("region columns require one contig, start, and stop/end alias")
    return tuple(
        _as_array(intervals[name])
        for name in (contig_names[0], start_names[0], stop_names[0])
    )


def _array_columns(intervals):
    if (
        isinstance(intervals, (tuple, list))
        and len(intervals) == 3
        and all(isinstance(column, np.ndarray) for column in intervals)
    ):
        return intervals
    return None


def _row_columns(intervals):
    try:
        rows = list(intervals)
    except TypeError as error:
        raise TypeError("regions must be a supported interval input") from error
    if not rows:
        raise ValueError("regions must not be empty")
    try:
        contigs, starts, stops = zip(*rows, strict=True)
    except (TypeError, ValueError) as error:
        raise TypeError("each region row must have contig, start, and stop") from error
    return np.asarray(contigs), np.asarray(starts), np.asarray(stops)


def _as_array(column) -> np.ndarray:
    to_numpy = getattr(column, "to_numpy", None)
    return np.asarray(to_numpy() if callable(to_numpy) else column)


def _validate_columns(contigs, starts, stops) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contigs = np.asarray(contigs)
    starts = _integer_column(starts, "start")
    stops = _integer_column(stops, "stop")
    if contigs.ndim != 1:
        raise ValueError("region contig column must be one-dimensional")
    if contigs.size != starts.size or starts.size != stops.size:
        raise ValueError("region columns must have equal lengths")
    if contigs.size == 0:
        raise ValueError("regions must not be empty")
    if contigs.dtype.kind == "O":
        if not all(isinstance(contig, str) for contig in contigs):
            raise TypeError("region contig column must contain strings")
        contigs = contigs.astype(str)
    if contigs.dtype.kind != "U" or np.any(contigs == ""):
        raise TypeError("region contig column must contain nonempty strings")
    return contigs, starts, stops


def _integer_column(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"region {name} column must be one-dimensional")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError(f"region {name} column must contain signed integers")
    if values.dtype.kind == "u" and np.any(values > _I64_MAX):
        raise ValueError(f"region {name} is outside signed 64-bit range")
    return values.astype(np.int64, copy=False)


def _contig_ids(contigs: np.ndarray, names: np.ndarray) -> np.ndarray:
    order = np.argsort(contigs, kind="stable")
    sorted_contigs = contigs[order]
    positions = np.searchsorted(sorted_contigs, names)
    found = positions < sorted_contigs.size
    found[found] = sorted_contigs[positions[found]] == names[found]
    if not np.all(found):
        raise KeyError(f"unknown contig {names[np.flatnonzero(~found)[0]]!r}")
    return order[positions].astype(np.int64, copy=False)


def _build_region_layout(
    contig_ids: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    flat_starts: np.ndarray,
    flat_stops: np.ndarray,
    input_index: np.ndarray,
) -> RegionLayout:
    lengths = stops - starts
    packed_offsets = np.empty(lengths.size + 1, dtype=np.int64)
    packed_offsets[0] = 0
    np.cumsum(lengths, out=packed_offsets[1:])
    arrays = (
        contig_ids,
        starts,
        stops,
        flat_starts,
        flat_stops,
        packed_offsets,
        input_index,
    )
    for array in arrays:
        array.setflags(write=False)
    return RegionLayout(*arrays)
