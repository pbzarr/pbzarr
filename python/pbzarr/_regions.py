"""Interval parsing and the batch engine behind ``reduce_regions``."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from dask import delayed
import dask.array as da
import numpy as np
import xarray as xr
import zarr

from . import _zip
from ._region import RegionQuery, parse_region

# Set by `open` on a track's Dask array: where the raw Zarr array lives, so
# region tasks can read it directly instead of through the source graph.
_PBZ_SOURCE_ATTR = "_pbz_source"

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
# One task holds at most about this many bytes of rows, unless one region is larger.
_BATCH_BYTES = 128 * 1024**2
# One task touches about this many Dask position chunks at most.
_BATCH_CHUNKS = 16

REDUCERS = ("sum", "mean", "min", "max", "count", "std", "var", "median", "quantile", "summit")
_OPTION_REDUCERS = {
    "q": {"quantile"},
    "by": {"summit"},
    "ddof": {"std", "var"},
    "skipna": {"sum", "mean", "min", "max", "std", "var", "median", "quantile"},
}


def reducer_options(reducer: str, **given) -> dict:
    """Check the reducer and its options, and fill in the defaults."""
    if not isinstance(reducer, str) or reducer not in REDUCERS:
        raise ValueError(f"unsupported reducer {reducer!r}; choose one of: {', '.join(REDUCERS)}")
    for name, value in given.items():
        if value is not None and reducer not in _OPTION_REDUCERS[name]:
            raise TypeError(f"reducer {reducer!r} does not take {name}=")
    q = given["q"]
    if reducer == "quantile":
        if q is None:
            raise TypeError("quantile needs q=")
        q = np.asarray(q, dtype=np.float64)
        if q.ndim > 1 or q.size == 0 or np.any(np.isnan(q)) or np.any((q < 0) | (q > 1)):
            raise ValueError("q must be a number or a list of numbers in [0, 1]")
    by = given["by"]
    if reducer == "summit":
        if isinstance(by, str):
            by = (by,)
        if (
            not isinstance(by, (tuple, list))
            or not by
            or any(not isinstance(name, str) or not name for name in by)
        ):
            raise TypeError("summit needs by=(variable, ...)")
        if len(set(by)) != len(by):
            raise ValueError("summit ranking variables must be unique")
        by = tuple(by)
    ddof = given["ddof"]
    if ddof is None:
        ddof = 0
    elif isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be an integer")
    elif ddof < 0:
        raise ValueError("ddof must be nonnegative")
    skipna = given["skipna"]
    return {
        "q": q,
        "by": by,
        "ddof": int(ddof),
        "skipna": True if skipna is None else bool(skipna),
    }


@dataclass(frozen=True)
class _Batch:
    """The regions of one task and the flat rows it reads for them."""

    regions: np.ndarray
    spans: np.ndarray
    sliced: bool
    starts: np.ndarray
    ends: np.ndarray

    def rows(self) -> slice | np.ndarray:
        return _rows(self.spans, self.sliced)


def _rows(spans: np.ndarray, sliced: bool) -> slice | np.ndarray:
    """One range over all spans, or the exact rows of the spans."""
    if sliced:
        return slice(int(spans[0, 0]), int(spans[-1, 1]))
    lengths = spans[:, 1] - spans[:, 0]
    edges = np.cumsum(lengths) - lengths
    return np.arange(int(lengths.sum()), dtype=np.int64) + np.repeat(spans[:, 0] - edges, lengths)


def _plan(
    flat_starts: np.ndarray,
    flat_ends: np.ndarray,
    row_bytes: int,
    boundaries: np.ndarray | None,
) -> list[_Batch]:
    """Batches of regions in position order, bounded by bytes and Dask chunks.

    Overlapping and adjacent regions share one span, so their rows are read
    once. A batch reads one range when its spans cover at least half of it,
    and the exact rows of the spans when they do not.
    """
    order = np.lexsort((flat_ends, flat_starts))
    starts = flat_starts[order]
    ends = flat_ends[order]
    reach = np.maximum.accumulate(ends)
    head = np.ones(order.size, dtype=bool)
    head[1:] = starts[1:] > reach[:-1]
    span_of = np.cumsum(head) - 1
    heads = np.flatnonzero(head)
    span_lo = starts[heads]
    span_hi = reach[np.append(heads[1:] - 1, order.size - 1)]

    span_bytes = (span_hi - span_lo) * max(row_bytes, 1)
    batch = (np.cumsum(span_bytes) - span_bytes) // _BATCH_BYTES
    if boundaries is not None:
        first_chunk = np.searchsorted(boundaries, span_lo, side="right") - 1
        last_chunk = np.searchsorted(boundaries, span_hi - 1, side="right") - 1
        seen = np.concatenate(([-1], last_chunk[:-1]))
        new_chunks = np.maximum(last_chunk - np.maximum(first_chunk, seen + 1) + 1, 0)
        batch = np.maximum(batch, (np.cumsum(new_chunks) - new_chunks) // _BATCH_CHUNKS)

    span_edges = np.concatenate(([0], np.flatnonzero(np.diff(batch)) + 1, [heads.size]))
    region_edges = np.append(heads, order.size)[span_edges]
    batches = []
    for index in range(span_edges.size - 1):
        a, b = int(span_edges[index]), int(span_edges[index + 1])
        r0, r1 = int(region_edges[index]), int(region_edges[index + 1])
        lo, hi = span_lo[a:b], span_hi[a:b]
        region_starts, region_ends = starts[r0:r1], ends[r0:r1]
        sliced = 2 * int((hi - lo).sum()) >= int(hi[-1] - lo[0])
        if sliced:
            local = region_starts - lo[0]
        else:
            span = span_of[r0:r1] - a
            edges = np.cumsum(hi - lo) - (hi - lo)
            local = edges[span] + (region_starts - lo[span])
        batches.append(
            _Batch(
                regions=order[r0:r1],
                spans=np.column_stack((lo, hi)),
                sliced=sliced,
                starts=local,
                ends=local + (region_ends - region_starts),
            )
        )
    return batches


@dataclass(frozen=True)
class _StoreRead:
    """Rows of a stored Zarr array, read inside the task."""

    source: dict
    spans: np.ndarray
    sliced: bool
    cols: slice | None

    def read(self) -> np.ndarray:
        kwargs = {}
        if self.source["storage_options"] is not None:
            kwargs["storage_options"] = self.source["storage_options"]
        location = self.source["source"]
        zipped = _zip.zip_path(location)
        store = _zip.open_store(zipped) if zipped is not None else None
        try:
            array = zarr.open_array(
                store if store is not None else location,
                mode="r",
                path=self.source["array_path"],
                **kwargs,
            )
            rows = _rows(self.spans, self.sliced)
            selection = (rows,) if self.cols is None else (rows, self.cols)
            if self.sliced:
                return np.asarray(array[selection])
            return np.asarray(array.get_orthogonal_selection(selection))
        finally:
            if store is not None:
                store.close()


def _store_source(track: xr.DataArray) -> dict | None:
    source = getattr(track.data, _PBZ_SOURCE_ATTR, None)
    if track.chunks is None or not isinstance(source, dict):
        return None
    if any(key not in source for key in ("source", "array_path", "storage_options")):
        return None
    return source


def _read(track: xr.DataArray, batch: _Batch, cols: slice | None):
    """The rows of one batch: a store read, a Dask slice, or a NumPy array."""
    source = _store_source(track)
    if source is not None:
        return _StoreRead(source, batch.spans, batch.sliced, cols)
    rows = batch.rows()
    key = (rows,) if cols is None else (rows, cols)
    if track.chunks is not None:
        return track.data[key]
    return np.asarray(track.variable[key].values)


def _reduce_batch(inputs, starts, ends, region_starts, reducer, options) -> tuple:
    arrays = [item.read() if isinstance(item, _StoreRead) else np.asarray(item) for item in inputs]
    arrays, offsets = _pack(arrays, np.asarray(starts), np.asarray(ends))
    if reducer == "summit":
        return _summit(arrays, offsets, np.asarray(region_starts), options["ranks"])
    return (_reduce(reducer, arrays[0], offsets, options),)


def _pack(arrays: list[np.ndarray], starts: np.ndarray, ends: np.ndarray):
    """Rows of each region next to each other, and the region edges in them."""
    lengths = ends - starts
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if offsets[-1] == arrays[0].shape[0] and np.array_equal(starts, offsets[:-1]):
        return arrays, offsets
    rows = np.arange(offsets[-1], dtype=np.int64) + np.repeat(starts - offsets[:-1], lengths)
    return [np.take(array, rows, axis=0) for array in arrays], offsets


def _sum_dtype(dtype: np.dtype) -> np.dtype:
    if dtype.kind in "bi":
        return np.dtype(np.int64)
    if dtype.kind == "u":
        return np.dtype(np.uint64)
    return dtype


def _reduce(reducer: str, values: np.ndarray, offsets: np.ndarray, options: dict) -> np.ndarray:
    heads = offsets[:-1]
    lengths = np.diff(offsets)
    floating = values.dtype.kind == "f"
    skip = floating and options["skipna"]
    if reducer in ("median", "quantile"):
        q = 0.5 if reducer == "median" else options["q"]
        return _quantiles(values, offsets, q, skip)
    if reducer in ("min", "max"):
        if skip:
            ufunc = np.fmin if reducer == "min" else np.fmax
        else:
            ufunc = np.minimum if reducer == "min" else np.maximum
        return ufunc.reduceat(values, heads, axis=0)

    missing = np.isnan(values) if skip else None
    if missing is None:
        shape = (lengths.size,) + values.shape[1:]
        counts = np.broadcast_to(lengths.reshape((-1,) + (1,) * (values.ndim - 1)), shape)
    else:
        counts = np.add.reduceat(~missing, heads, axis=0, dtype=np.int64)
    if reducer == "count":
        return np.ascontiguousarray(counts, dtype=np.int64)
    clean = np.where(missing, 0, values) if skip else values
    total = np.add.reduceat(clean, heads, axis=0, dtype=_sum_dtype(values.dtype))
    if reducer == "sum":
        return total

    real = values.dtype if floating else np.dtype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total / counts).astype(real)
        if reducer == "mean":
            return mean
        deviation = values - np.repeat(mean, lengths, axis=0)
        square = deviation * deviation
        if skip:
            square[missing] = 0
        dof = counts - options["ddof"]
        var = np.where(dof > 0, np.add.reduceat(square, heads, axis=0) / dof, np.nan)
    var = var.astype(real)
    return var if reducer == "var" else np.sqrt(var)


def _quantiles(values: np.ndarray, offsets: np.ndarray, q, skip: bool) -> np.ndarray:
    function = np.nanquantile if skip else np.quantile
    q = np.asarray(q, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parts = [
            function(values[a:b], q, axis=0)
            for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())
        ]
    result = np.stack(parts)
    return np.moveaxis(result, 1, -1) if q.ndim == 1 else result


def _summit(arrays: list[np.ndarray], offsets: np.ndarray, region_starts: np.ndarray, ranks):
    """At each region and column, the values at the row with the highest ``by`` keys.

    Keys compare in order; ties go to the first row. NaN ranks lowest.
    """
    heads = offsets[:-1]
    lengths = np.diff(offsets)
    first = arrays[0]
    candidates = np.ones(first.shape, dtype=bool)
    for index in ranks:
        key = arrays[index]
        if key.dtype.kind == "f":
            key = np.where(np.isnan(key), -np.inf, key)
            floor = -np.inf
        elif key.dtype.kind == "i":
            floor = np.iinfo(key.dtype).min
        else:
            floor = 0 if key.dtype.kind == "u" else False
        maxima = np.maximum.reduceat(np.where(candidates, key, floor), heads, axis=0)
        candidates &= key == np.repeat(maxima, lengths, axis=0)

    n = first.shape[0]
    column_shape = (1,) * (first.ndim - 1)
    row_ids = np.broadcast_to(np.arange(n, dtype=np.int64).reshape((n, *column_shape)), first.shape)
    winners = np.minimum.reduceat(np.where(candidates, row_ids, n), heads, axis=0)
    tail = tuple(np.indices(winners.shape, sparse=False)[1:])
    selected = tuple(array[(winners, *tail)] for array in arrays)
    shape = (heads.size, *column_shape)
    positions = region_starts.reshape(shape) + winners - heads.reshape(shape)
    return (*selected, positions.astype(np.int64))


def _column_blocks(track: xr.DataArray) -> list[slice | None]:
    if track.ndim == 1:
        return [None]
    if track.chunks is None:
        return [slice(0, track.shape[1])]
    edges = np.concatenate(([0], np.cumsum(track.chunks[1])))
    return [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _row_bytes(track: xr.DataArray) -> int:
    if track.ndim == 1:
        return track.dtype.itemsize
    width = max(track.chunks[1]) if track.chunks is not None else track.shape[1]
    return track.dtype.itemsize * int(width)


def _output_dtypes(group: list[xr.DataArray], reducer: str, options: dict) -> list[np.dtype]:
    dummy = [np.zeros((2,) + (1,) * (track.ndim - 1), dtype=track.dtype) for track in group]
    zero = np.zeros(1, dtype=np.int64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _reduce_batch(dummy, zero, zero + 2, zero, reducer, options)
    return [output.dtype for output in outputs]


def _summit_ranks(tracks: list[xr.DataArray], by: tuple[str, ...]) -> tuple[int, ...]:
    names = [track.name for track in tracks]
    missing = [name for name in by if name not in names]
    if missing:
        raise ValueError("unknown summit ranking variable(s): " + ", ".join(map(str, missing)))
    first = tracks[0]
    if any(track.dims != first.dims or track.shape != first.shape for track in tracks[1:]):
        raise ValueError("summit needs all variables to have the same dimensions and shape")
    if any(tracks[names.index(name)].dtype.kind not in "biuf" for name in by):
        raise TypeError("summit ranking variables must be numeric")
    return tuple(names.index(name) for name in by)


def reduce_regions_dataset(
    ds: xr.Dataset, reference, intervals, reducer: str, options: dict
) -> xr.Dataset:
    """Reduce every position-first track of ``ds`` over each interval.

    The result has one row per interval, in input order, and the coordinates
    ``region_contig``, ``region_start`` and ``region_end``.
    """
    contig_ids, starts, ends = interval_columns(reference, intervals)
    base = reference.offsets[contig_ids]
    tracks = [ds[name] for name in ds.data_vars]
    lazy = any(track.chunks is not None for track in tracks)
    if lazy:
        tracks = [track if track.chunks is not None else track.chunk() for track in tracks]
    summit = reducer == "summit"
    if summit:
        options = {**options, "ranks": _summit_ranks(tracks, options["by"])}
        groups = [tracks]
    else:
        groups = [[track] for track in tracks]

    widths = [_row_bytes(track) for track in tracks]
    row_bytes = sum(widths) if summit else max(widths)
    boundaries = np.concatenate(([0], np.cumsum(tracks[0].chunks[0]))) if lazy else None
    batches = _plan(base + starts, base + ends, row_bytes, boundaries)
    order = np.concatenate([batch.regions for batch in batches])
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    in_order = bool(np.array_equal(order, np.arange(order.size)))
    stack = da.concatenate if lazy else np.concatenate

    vector_q = reducer == "quantile" and options["q"].ndim == 1
    results = {}
    for group in groups:
        dtypes = _output_dtypes(group, reducer, options)
        blocks = _column_blocks(group[0]) if lazy else [None]
        outputs = [[] for _ in dtypes]
        for batch in batches:
            region_starts = starts[batch.regions] if summit else None
            pieces = [[] for _ in dtypes]
            for cols in blocks:
                inputs = [_read(track, batch, cols) for track in group]
                args = (inputs, batch.starts, batch.ends, region_starts, reducer, options)
                if not lazy:
                    for piece, value in zip(pieces, _reduce_batch(*args)):
                        piece.append(value)
                    continue
                shape = (batch.regions.size,)
                shape += group[0].shape[1:] if cols is None else (cols.stop - cols.start,)
                if vector_q:
                    shape += (options["q"].size,)
                task = delayed(_reduce_batch, nout=len(dtypes))(*args)
                for i, dtype in enumerate(dtypes):
                    pieces[i].append(da.from_delayed(task[i], shape=shape, dtype=dtype))
            for output, columns in zip(outputs, pieces):
                output.append(columns[0] if len(columns) == 1 else stack(columns, axis=1))
        joined = [rows[0] if len(rows) == 1 else stack(rows, axis=0) for rows in outputs]
        if not in_order:
            joined = [data[rank] for data in joined]
        dims = ("region", *group[0].dims[1:])
        for track, data in zip(group, joined):
            results[track.name] = xr.Variable(
                (*dims, "quantile") if vector_q else dims,
                data,
                attrs=track.attrs if summit else {},
            )
        if summit:
            summit_positions = xr.Variable(dims, joined[-1])

    names = np.asarray(reference.names, dtype=object)
    coords = {
        "region_contig": xr.Variable("region", names[contig_ids]),
        "region_start": xr.Variable("region", starts),
        "region_end": xr.Variable("region", ends),
    }
    if summit:
        coords["summit_position"] = summit_positions
    result = xr.Dataset(results, coords=xr.Coordinates(coords, indexes={}))
    if reducer == "quantile":
        q = options["q"]
        result = result.assign_coords(quantile=("quantile", q) if vector_q else q.item())
    return result


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
            "(contig, start, end) tuple"
        )

    if not isinstance(normalized.contig, str) or not normalized.contig:
        raise ValueError("region contig must be a nonempty string")
    return RegionQuery(
        normalized.contig,
        _normalize_coordinate(normalized.start, "start"),
        _normalize_coordinate(normalized.end, "end"),
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


def _resolve_one(
    reference, normalized: RegionQuery
) -> tuple[RegionQuery, slice]:
    """One region's contig-local bounds and flat rows."""
    resolved = reference.resolve(normalized)
    return resolved, reference.rows(resolved)


def interval_columns(reference, intervals) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contig indexes and contig-local starts and ends of ``intervals``, in input order."""
    if isinstance(intervals, (RegionQuery, str)) or (
        isinstance(intervals, tuple)
        and len(intervals) == 3
        and isinstance(intervals[0], str)
        and all(
            bound is None
            or (isinstance(bound, (int, np.integer)) and not isinstance(bound, bool))
            for bound in intervals[1:]
        )
    ):
        resolved, _ = _resolve_one(reference, _normalize_one(intervals))
        return (
            np.asarray([reference.names.index(resolved.contig)], dtype=np.int64),
            np.asarray([resolved.start], dtype=np.int64),
            np.asarray([resolved.end], dtype=np.int64),
        )

    columns = _dataframe_columns(intervals)
    if columns is None:
        columns = _array_columns(intervals)
    if columns is None:
        columns = _query_columns(intervals, reference)
    if columns is None:
        columns = _row_columns(intervals)
    contig_names, starts, stops = _validate_columns(*columns)

    contigs = np.asarray(reference.names, dtype=object)
    contig_ids = _contig_ids(contigs, contig_names)
    offsets = reference.offsets
    contig_lengths = offsets[contig_ids + 1] - offsets[contig_ids]
    if np.any(starts < 0):
        raise ValueError("region start must be nonnegative")
    if np.any(stops <= starts):
        raise ValueError("region must be nonempty with start less than end")
    if np.any(stops > contig_lengths):
        raise ValueError("region end exceeds contig length")
    return contig_ids, starts, stops


def _query_columns(intervals, reference):
    """A list of region strings or RegionQuery objects, as three columns."""
    if not isinstance(intervals, (list, tuple)) or not intervals or not all(
        isinstance(item, (str, RegionQuery)) for item in intervals
    ):
        return None
    resolved = [_resolve_one(reference, _normalize_one(item))[0] for item in intervals]
    return (
        np.asarray([r.contig for r in resolved], dtype=object),
        np.asarray([r.start for r in resolved], dtype=np.int64),
        np.asarray([r.end for r in resolved], dtype=np.int64),
    )


def _dataframe_columns(intervals):
    if not (hasattr(intervals, "columns") and hasattr(intervals, "__getitem__")):
        return None
    names = list(intervals.columns)
    contig_names = [name for name in names if name in {"contig", "chrom", "#chrom"}]
    start_names = [name for name in names if name == "start"]
    end_names = [name for name in names if name == "end"]
    if len(contig_names) != 1 or len(start_names) != 1 or len(end_names) != 1:
        raise ValueError("region columns require one contig, start, and end column")
    return tuple(
        _as_array(intervals[name])
        for name in (contig_names[0], start_names[0], end_names[0])
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
        raise TypeError("each region row must have contig, start, and end") from error
    return np.asarray(contigs), np.asarray(starts), np.asarray(stops)


def _as_array(column) -> np.ndarray:
    to_numpy = getattr(column, "to_numpy", None)
    return np.asarray(to_numpy() if callable(to_numpy) else column)


def _validate_columns(contigs, starts, stops) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contigs = np.asarray(contigs)
    starts = _integer_column(starts, "start")
    stops = _integer_column(stops, "end")
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
    if contigs.dtype != names.dtype:
        # Stores opened under numpy 2 hold contigs as StringDType while
        # query names arrive fixed-width; searchsorted refuses to mix them.
        names = names.astype(contigs.dtype)
    order = np.argsort(contigs, kind="stable")
    sorted_contigs = contigs[order]
    positions = np.searchsorted(sorted_contigs, names)
    found = positions < sorted_contigs.size
    found[found] = sorted_contigs[positions[found]] == names[found]
    if not np.all(found):
        raise KeyError(f"unknown contig {names[np.flatnonzero(~found)[0]]!r}")
    return order[positions].astype(np.int64, copy=False)


