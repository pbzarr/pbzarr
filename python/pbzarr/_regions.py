"""Normalization and geometry resolution for PBZ region operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading

import dask
import dask.array as da
from dask._task_spec import Task
from dask.base import tokenize
from dask import delayed
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
import numpy as np
import xarray as xr
import zarr

from ._native import PbzError
from ._region import RegionQuery, parse_region
from ._xarray import _PBZ_SOURCE_ATTR, _validate_track


_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_I32_MAX = 2**31 - 1
_TARGET_BYTES = 128 * 1024**2
_MAX_SOURCE_BLOCKS = 16
_PACKED_REDUCERS = frozenset(
    {
        "mean",
        "sum",
        "min",
        "max",
        "count",
        "std",
        "var",
        "median",
        "quantile",
        "summit",
    }
)
_PACKED_REGION_COORDS = (
    "region_contig",
    "region_start",
    "region_stop",
    "region_input_index",
    "region_storage_index",
)
_OPEN_ARRAYS = {}
_OPEN_ARRAYS_LOCK = threading.Lock()


def _contains_task_function(value, function) -> bool:
    if isinstance(value, Task):
        if value.func is function:
            return True
        return any(
            _contains_task_function(item, function)
            for item in (*value.args, *value.kwargs.values())
        )
    if isinstance(value, dict):
        return any(
            _contains_task_function(item, function) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_task_function(item, function) for item in value)
    return False


def _fuse_region_reduction_tasks(values: da.Array) -> da.Array:
    """Fuse each single-consumer PBZ gather into its reduction task."""
    (optimized,) = dask.optimize(values)
    graph = optimized.__dask_graph__().to_dict()
    dependents: dict[object, set[object]] = {}
    for consumer_key, node in graph.items():
        for dependency in getattr(node, "dependencies", ()):
            dependents.setdefault(dependency, set()).add(consumer_key)

    fused = False
    for key, node in list(graph.items()):
        if not isinstance(node, Task) or not any(
            _contains_task_function(node, function)
            for function in (_gather_block, _read_gather_block)
        ):
            continue
        producer_keys = [
            dependency
            for dependency in node.dependencies
            if isinstance(graph.get(dependency), Task)
        ]
        if (
            not producer_keys
            or len(producer_keys) > _MAX_SOURCE_BLOCKS
            or any(dependents.get(producer) != {key} for producer in producer_keys)
        ):
            continue
        producers = [graph[producer] for producer in producer_keys]
        graph[key] = Task.fuse(*producers, node, key=key)
        for producer in producer_keys:
            del graph[producer]
        fused = True

    if not fused:
        return optimized
    layer_name = f"pbz-region-reduction-{optimized.name}"
    high_level_graph = HighLevelGraph(
        {layer_name: MaterializedLayer(graph)},
        {layer_name: set()},
    )
    meta = np.empty((0,) * optimized.ndim, dtype=optimized.dtype)
    return da.Array(
        high_level_graph,
        optimized.name,
        optimized.chunks,
        dtype=optimized.dtype,
        meta=meta,
    )


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


def _validate_packed_dataset(ds: xr.Dataset) -> np.ndarray:
    """Validate the packed contract and return read-only offsets."""
    if ds.attrs.get("pbz:representation") != "packed-regions":
        raise ValueError(
            "reduce requires a Dataset with pbz:representation='packed-regions'"
        )
    if not ds.data_vars:
        raise ValueError("packed regions must contain at least one data variable")
    if "position" not in ds.sizes or "region" not in ds.sizes:
        raise ValueError("packed regions must have position and region dimensions")
    if "position" in ds.indexes or "region" in ds.indexes:
        raise ValueError("packed position and region dimensions must be unindexed")

    required = {"offsets", *_PACKED_REGION_COORDS}
    missing = sorted(required.difference(ds.coords))
    if missing:
        raise ValueError(
            "packed regions are missing required coordinates: "
            + ", ".join(missing)
        )
    if ds["offsets"].dims != ("region_boundary",):
        raise ValueError("offsets must have only the region_boundary dimension")
    for name in _PACKED_REGION_COORDS:
        if ds[name].dims != ("region",):
            raise ValueError(f"{name} must have only the region dimension")

    offsets_data = np.asarray(ds["offsets"].data)
    if offsets_data.dtype.kind not in "iu":
        raise ValueError("offsets must contain integers")
    if not np.can_cast(offsets_data.dtype, np.dtype(np.int64), casting="safe"):
        raise ValueError("offsets must fit in signed 64-bit integers")
    offsets = offsets_data.astype(np.int64, copy=True)
    n_regions = ds.sizes["region"]
    if offsets.size != n_regions + 1:
        raise ValueError("offsets must have one more element than region")
    if offsets[0] != 0:
        raise ValueError("offsets must start at zero")
    if offsets[-1] != ds.sizes["position"]:
        raise ValueError("offsets must end at the packed position size")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must be strictly increasing")

    integer_coordinates = (
        "region_start",
        "region_stop",
        "region_input_index",
        "region_storage_index",
    )
    normalized = {}
    for name in integer_coordinates:
        data = np.asarray(ds[name].data)
        if data.dtype.kind not in "iu" or not np.can_cast(
            data.dtype, np.dtype(np.int64), casting="safe"
        ):
            raise ValueError(f"{name} must contain integers that fit in int64")
        normalized[name] = data.astype(np.int64, copy=False)

    starts = normalized["region_start"]
    stops = normalized["region_stop"]
    if np.any(stops <= starts):
        raise ValueError("packed regions must have positive lengths")
    if not np.array_equal(stops - starts, np.diff(offsets)):
        raise ValueError("region lengths must match packed offsets")
    storage_order = np.arange(n_regions, dtype=np.int64)
    if not np.array_equal(normalized["region_storage_index"], storage_order):
        raise ValueError("region_storage_index must equal packed storage order")
    if not np.array_equal(
        np.sort(normalized["region_input_index"]), storage_order
    ):
        raise ValueError("region_input_index must be a permutation of region indices")

    contigs = np.asarray(ds["region_contig"].data)
    if contigs.dtype.kind not in "OUT" or any(
        not isinstance(value, str) or not value for value in contigs.tolist()
    ):
        raise ValueError("region_contig must contain non-empty strings")

    for name, values in ds.data_vars.items():
        if values.ndim not in {1, 2} or values.dims[0] != "position":
            raise ValueError(f"{name} must be a scalar or 2D position-first variable")
        if values.sizes["position"] != offsets[-1]:
            raise ValueError(f"{name} does not cover the packed position axis")
        if any("position" in coordinate.dims for coordinate in values.coords.values()):
            raise ValueError(
                f"{name} cannot have position-dependent coordinates"
            )
        if values.chunks is not None:
            boundaries = np.cumsum(values.chunks[0], dtype=np.int64)
            if not np.all(np.isin(boundaries, offsets)):
                raise ValueError(
                    f"{name} position chunks must end at complete-region boundaries"
                )

    offsets.setflags(write=False)
    return offsets


def _reduce_block(
    block,
    local_offsets,
    dimensions,
    attrs,
    reducer,
    kwargs,
) -> np.ndarray:
    local_values = xr.DataArray(np.asarray(block), dims=dimensions, attrs=attrs)
    labels = np.repeat(
        np.arange(len(local_offsets) - 1, dtype=np.int64),
        np.diff(local_offsets),
    )
    groups = xr.DataArray(labels, dims=("position",), name="region")
    method = getattr(local_values.groupby(groups), reducer)
    return np.asarray(method(dim="position", **kwargs).data)


def _reduction_template(
    values: xr.DataArray, reducer: str, kwargs: dict
) -> xr.DataArray:
    shape = (2,) + (1,) * (values.ndim - 1)
    dummy = xr.DataArray(
        np.zeros(shape, dtype=values.dtype),
        dims=values.dims,
        attrs=values.attrs,
    )
    groups = xr.DataArray(
        np.zeros(2, dtype=np.int64), dims=("position",), name="region"
    )
    method = getattr(dummy.groupby(groups), reducer)
    return method(dim="position", **kwargs)


def _validate_reducer(reducer: str, kwargs: dict) -> None:
    if not isinstance(reducer, str) or reducer not in _PACKED_REDUCERS:
        allowed = ", ".join(sorted(_PACKED_REDUCERS))
        raise ValueError(f"unsupported reducer {reducer!r}; choose one of: {allowed}")
    if "dim" in kwargs:
        raise TypeError("reduce_regions controls dim='position'; do not pass dim")
    if reducer == "summit":
        unexpected = sorted(set(kwargs) - {"by"})
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"summit got unexpected keyword argument(s): {names}")
        by = kwargs.get("by")
        if (
            not isinstance(by, (tuple, list))
            or not by
            or any(not isinstance(name, str) or not name for name in by)
        ):
            raise TypeError("summit requires by=(variable, ...)")
        if len(set(by)) != len(by):
            raise ValueError("summit ranking variables must be unique")


def _summit_block(source_blocks, local_offsets, region_starts, ranking_indices):
    blocks = [np.asarray(block) for block in source_blocks]
    if not blocks or any(block.shape != blocks[0].shape for block in blocks[1:]):
        raise ValueError("summit variables must have matching shapes")

    starts = np.asarray(local_offsets[:-1], dtype=np.int64)
    lengths = np.diff(local_offsets).astype(np.int64, copy=False)
    candidates = np.ones(blocks[0].shape, dtype=bool)
    for index in ranking_indices:
        key = blocks[index]
        if key.dtype.kind == "f":
            comparable = np.where(np.isnan(key), -np.inf, key)
            minimum = -np.inf
        elif key.dtype.kind == "i":
            comparable = key
            minimum = np.iinfo(key.dtype).min
        elif key.dtype.kind == "u":
            comparable = key
            minimum = 0
        elif key.dtype.kind == "b":
            comparable = key
            minimum = False
        else:
            raise TypeError("summit ranking variables must be numeric")
        maxima = np.maximum.reduceat(
            np.where(candidates, comparable, minimum), starts, axis=0
        )
        candidates &= comparable == np.repeat(maxima, lengths, axis=0)

    row_shape = (blocks[0].shape[0],) + (1,) * (blocks[0].ndim - 1)
    row_ids = np.broadcast_to(
        np.arange(blocks[0].shape[0], dtype=np.int64).reshape(row_shape),
        blocks[0].shape,
    )
    winners = np.minimum.reduceat(
        np.where(candidates, row_ids, blocks[0].shape[0]), starts, axis=0
    )
    tail_indices = tuple(np.indices(winners.shape, sparse=False)[1:])
    selected = tuple(block[(winners, *tail_indices)] for block in blocks)
    start_shape = (len(region_starts),) + (1,) * (winners.ndim - 1)
    summit_positions = (
        np.asarray(region_starts, dtype=np.int64).reshape(start_shape)
        + winners
        - starts.reshape(start_shape)
    )
    return (*selected, summit_positions)


def _summit_packed_dataset(
    ds: xr.Dataset, offsets: np.ndarray, by
) -> xr.Dataset:
    names = list(ds.data_vars)
    missing = [name for name in by if name not in ds.data_vars]
    if missing:
        raise ValueError(
            "unknown summit ranking variable(s): " + ", ".join(missing)
        )

    first = ds[names[0]]
    if any(ds[name].dims != first.dims for name in names[1:]):
        raise ValueError("summit requires all variables to have matching dimensions")
    if any(ds[name].shape != first.shape for name in names[1:]):
        raise ValueError("summit requires all variables to have matching shapes")
    if any(ds[name].dtype.kind not in "biuf" for name in by):
        raise TypeError("summit ranking variables must be numeric")

    ranking_indices = tuple(names.index(name) for name in by)
    region_starts = np.asarray(ds["region_start"].data, dtype=np.int64)
    output_dims = ("region", *first.dims[1:])
    chunks = [ds[name].chunks for name in names]
    if any((chunk is None) != (chunks[0] is None) for chunk in chunks[1:]):
        raise ValueError("summit requires all variables to use the same array type")
    if chunks[0] is not None and any(chunk != chunks[0] for chunk in chunks[1:]):
        raise ValueError("summit requires all variables to have matching chunks")

    if chunks[0] is None:
        outputs = _summit_block(
            [ds[name].data for name in names],
            offsets,
            region_starts,
            ranking_indices,
        )
    else:
        source_blocks = [ds[name].data.to_delayed() for name in names]
        position_boundaries = np.empty(len(chunks[0][0]) + 1, dtype=np.int64)
        position_boundaries[0] = 0
        np.cumsum(chunks[0][0], out=position_boundaries[1:])
        region_edges = np.searchsorted(offsets, position_boundaries)
        rows = [[] for _ in range(len(names) + 1)]
        for position_block in range(len(chunks[0][0])):
            start_region = int(region_edges[position_block])
            stop_region = int(region_edges[position_block + 1])
            local_offsets = _local_offsets(
                offsets,
                start_region,
                stop_region,
                int(position_boundaries[position_block]),
            )
            column_blocks = (
                range(len(chunks[0][1])) if first.ndim == 2 else range(1)
            )
            block_outputs = [[] for _ in range(len(names) + 1)]
            for column_block in column_blocks:
                source_key = (
                    (position_block, column_block)
                    if first.ndim == 2
                    else (position_block,)
                )
                task = delayed(_summit_block)(
                    [blocks[source_key] for blocks in source_blocks],
                    local_offsets,
                    region_starts[start_region:stop_region],
                    ranking_indices,
                )
                output_shape = (stop_region - start_region,)
                if first.ndim == 2:
                    output_shape += (int(chunks[0][1][column_block]),)
                for output_index, name in enumerate(names):
                    block_outputs[output_index].append(
                        da.from_delayed(
                            task[output_index],
                            shape=output_shape,
                            dtype=ds[name].dtype,
                        )
                    )
                block_outputs[-1].append(
                    da.from_delayed(
                        task[-1], shape=output_shape, dtype=np.int64
                    )
                )
            for output_index, columns in enumerate(block_outputs):
                rows[output_index].append(
                    columns[0]
                    if len(columns) == 1
                    else da.concatenate(columns, axis=1)
                )
        outputs = tuple(
            _fuse_region_reduction_tasks(
                row[0] if len(row) == 1 else da.concatenate(row, axis=0)
            )
            for row in rows
        )

    data_variables = {
        name: xr.Variable(output_dims, outputs[index], attrs=ds[name].attrs)
        for index, name in enumerate(names)
    }
    coordinates = {
        name: coordinate.variable
        for name, coordinate in ds.coords.items()
        if name != "offsets"
        and "position" not in coordinate.dims
        and "region_boundary" not in coordinate.dims
    }
    coordinates["summit_position"] = xr.Variable(output_dims, outputs[-1])
    result = xr.Dataset(
        data_vars=data_variables,
        coords=xr.Coordinates(coordinates, indexes={}),
        attrs={
            key: value
            for key, value in ds.attrs.items()
            if not key.startswith("pbz:")
        },
    )
    result.set_close(ds.close)
    return result


def _local_offsets(offsets, start_region, stop_region, position_start):
    local = offsets[start_region : stop_region + 1] - position_start
    if int(local[-1]) <= _I32_MAX:
        local = local.astype(np.int32)
    local.setflags(write=False)
    return local


def _reduce_packed_dataset(
    ds: xr.Dataset, reducer: str, **kwargs
) -> xr.Dataset:
    offsets = _validate_packed_dataset(ds)
    _validate_reducer(reducer, kwargs)
    if reducer == "summit":
        return _summit_packed_dataset(ds, offsets, kwargs["by"])
    dask_region_batches = {}

    def eager_batches(values: xr.DataArray, template: xr.DataArray):
        bytes_per_position = np.dtype(values.dtype).itemsize
        for size in values.shape[1:]:
            bytes_per_position *= size
        region_edges = [0]
        batch_bytes = 0
        batch_start = 0
        for region_index, length in enumerate(np.diff(offsets)):
            region_bytes = int(length) * bytes_per_position
            if region_index > batch_start and batch_bytes + region_bytes > _TARGET_BYTES:
                region_edges.append(region_index)
                batch_start = region_index
                batch_bytes = 0
            batch_bytes += region_bytes
        region_edges.append(len(offsets) - 1)
        blocks = []
        for start_region, stop_region in zip(
            region_edges[:-1], region_edges[1:], strict=True
        ):
            start = int(offsets[start_region])
            stop = int(offsets[stop_region])
            local_offsets = offsets[start_region : stop_region + 1] - start
            blocks.append(
                _reduce_block(
                    values.data[start:stop],
                    local_offsets,
                    values.dims,
                    values.attrs,
                    reducer,
                    kwargs,
                )
            )
        return np.concatenate(blocks, axis=template.get_axis_num("region"))

    def dask_batches(values: xr.DataArray, template: xr.DataArray):
        source_blocks = values.data.to_delayed()
        batch_key = values.chunks[0]
        batches = dask_region_batches.get(batch_key)
        if batches is None:
            position_boundaries = np.empty(
                len(values.chunks[0]) + 1, dtype=np.int64
            )
            position_boundaries[0] = 0
            np.cumsum(values.chunks[0], out=position_boundaries[1:])
            region_edges = np.searchsorted(offsets, position_boundaries)
            batches = tuple(
                (
                    int(region_edges[position_block]),
                    int(region_edges[position_block + 1]),
                    _local_offsets(
                        offsets,
                        int(region_edges[position_block]),
                        int(region_edges[position_block + 1]),
                        int(position_boundaries[position_block]),
                    ),
                )
                for position_block in range(len(values.chunks[0]))
            )
            dask_region_batches[batch_key] = batches
        output_dims = template.dims
        region_axis = output_dims.index("region")
        rows = []
        for position_block, (
            start_region,
            stop_region,
            local_offsets,
        ) in enumerate(batches):
            region_size = stop_region - start_region
            columns = []
            column_blocks = (
                range(len(values.chunks[1]))
                if values.ndim == 2
                else range(1)
            )
            for column_block in column_blocks:
                source_key = (
                    (position_block, column_block)
                    if values.ndim == 2
                    else (position_block,)
                )
                output_shape = []
                for dimension in output_dims:
                    if dimension == "region":
                        output_shape.append(region_size)
                    elif values.ndim == 2 and dimension == values.dims[1]:
                        output_shape.append(int(values.chunks[1][column_block]))
                    else:
                        output_shape.append(template.sizes[dimension])
                task = delayed(_reduce_block)(
                    source_blocks[source_key],
                    local_offsets,
                    values.dims,
                    values.attrs,
                    reducer,
                    kwargs,
                )
                columns.append(
                    da.from_delayed(
                        task,
                        shape=tuple(output_shape),
                        dtype=template.dtype,
                    )
                )
            if len(columns) == 1:
                rows.append(columns[0])
            else:
                rows.append(
                    da.concatenate(
                        columns, axis=output_dims.index(values.dims[1])
                    )
                )
        reduced = da.concatenate(rows, axis=region_axis)
        return _fuse_region_reduction_tasks(reduced)

    data_variables = {}
    generated_coordinates = {}
    for name, values in ds.data_vars.items():
        template = _reduction_template(values, reducer, kwargs)
        if values.chunks is None:
            reduced = eager_batches(values, template)
        else:
            reduced = dask_batches(values, template)
        data_variables[name] = xr.Variable(
            template.dims,
            reduced,
            attrs=template.attrs,
        )
        for coordinate_name, coordinate in template.coords.items():
            if coordinate_name != "region":
                generated_coordinates[coordinate_name] = coordinate.variable

    coordinates = {}
    for name, coordinate in ds.coords.items():
        if name == "offsets" or "position" in coordinate.dims:
            continue
        if "region_boundary" in coordinate.dims:
            continue
        coordinates[name] = coordinate.variable
    coordinates.update(generated_coordinates)
    result = xr.Dataset(
        data_vars=data_variables,
        coords=xr.Coordinates(coordinates, indexes={}),
        attrs={
            key: value
            for key, value in ds.attrs.items()
            if not key.startswith("pbz:")
        },
    )
    result.set_close(ds.close)
    return result


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
    component_edges = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(first_blocks[1:] > last_blocks[:-1]) + 1,
            np.array([layout.n_regions], dtype=np.int64),
        )
    )
    region_edges = np.empty(layout.n_regions + 1, dtype=np.int64)
    region_edges[0] = 0
    edge_count = 1
    batch_bytes = 0
    batch_source_blocks = 0
    batch_start = 0
    for component_start, component_stop in zip(
        component_edges[:-1], component_edges[1:], strict=True
    ):
        component_start = int(component_start)
        component_stop = int(component_stop)
        next_bytes = (
            int(
                layout.packed_offsets[component_stop]
                - layout.packed_offsets[component_start]
            )
            * bytes_per_position
        )
        next_source_blocks = (
            int(last_blocks[component_stop - 1])
            - int(first_blocks[component_start])
            + 1
        )
        if component_start > batch_start and (
            batch_bytes + next_bytes > target_bytes
            or batch_source_blocks + next_source_blocks > max_source_blocks
        ):
            region_edges[edge_count] = component_start
            edge_count += 1
            batch_start = component_start
            batch_bytes = 0
            batch_source_blocks = 0
        batch_bytes += next_bytes
        batch_source_blocks += next_source_blocks
    region_edges[edge_count] = layout.n_regions
    edge_count += 1
    region_edges = region_edges[:edge_count]
    piece_edges = region_piece_edges[region_edges]

    largest_bound = max(
        layout.n_regions,
        n_pieces,
        len(position_chunks),
        max(position_chunks, default=0),
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
    piece_block_starts = np.empty(n_pieces, dtype=bool)
    piece_block_starts[0] = True
    piece_block_starts[1:] = piece_blocks[1:] != piece_blocks[:-1]
    batch_source_block_counts = np.add.reduceat(
        piece_block_starts.astype(np.int64, copy=False),
        piece_edges[:-1],
    )
    within_limits = (batch_positions <= max_batch_positions) & (
        batch_source_block_counts <= max_source_blocks
    )
    legal_region_boundaries = np.empty(layout.n_regions + 1, dtype=bool)
    legal_region_boundaries[0] = True
    legal_region_boundaries[-1] = True
    legal_region_boundaries[1:-1] = first_blocks[1:] > last_blocks[:-1]
    unsplittable = np.fromiter(
        (
            not np.any(legal_region_boundaries[start + 1 : stop])
            for start, stop in zip(
                region_edges[:-1], region_edges[1:], strict=True
            )
        ),
        dtype=bool,
        count=len(region_edges) - 1,
    )
    def has_valid_plan() -> bool:
        return (
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
            and np.all(piece_block_starts[piece_edges[:-1]])
            and np.array_equal(
                np.add.reduceat(piece_lengths, region_piece_edges[:-1]),
                np.diff(layout.packed_offsets),
            )
            and int(piece_lengths.sum()) == layout.total_positions
            and np.all(within_limits | unsplittable)
        )

    if not has_valid_plan():
        raise AssertionError("invalid packed region plan")

    for array in (batch_region_edges, batch_piece_edges, pieces):
        array.setflags(write=False)
    return batch_region_edges, batch_piece_edges, pieces


def _gather_block(source_blocks, pieces, output_shape) -> np.ndarray:
    if not source_blocks:
        raise ValueError("a gather block must contain at least one piece")

    first = np.asarray(source_blocks[0])
    output = np.empty(output_shape, dtype=first.dtype)
    block_ids = pieces["source_block"]
    if np.any(block_ids < 0) or np.any(block_ids >= len(source_blocks)):
        raise ValueError("piece references an unknown source block")

    starts = pieces["source_start"].astype(np.int64, copy=False)
    lengths = (
        pieces["source_stop"].astype(np.int64, copy=False) - starts
    )
    destinations = np.empty(len(pieces) + 1, dtype=np.int64)
    destinations[0] = 0
    np.cumsum(lengths, out=destinations[1:])
    if int(destinations[-1]) != output_shape[0]:
        raise ValueError("piece lengths do not cover the gather output")
    if len(pieces) == 0:
        return output

    run_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1,
        )
    )
    run_stops = np.concatenate(
        (run_starts[1:], np.array([len(pieces)], dtype=np.int64))
    )
    for piece_start, piece_stop in zip(run_starts, run_stops, strict=True):
        piece_start = int(piece_start)
        piece_stop = int(piece_stop)
        source = np.asarray(source_blocks[int(block_ids[piece_start])])
        output_start = int(destinations[piece_start])
        output_stop = int(destinations[piece_stop])
        run_size = output_stop - output_start
        index_dtype = (
            np.int32
            if max(source.shape[0], run_size) <= _I32_MAX
            else np.int64
        )
        relative_destinations = (
            destinations[piece_start:piece_stop] - output_start
        )
        bases = (
            starts[piece_start:piece_stop] - relative_destinations
        ).astype(index_dtype, copy=False)
        indices = np.arange(run_size, dtype=index_dtype)
        indices += np.repeat(bases, lengths[piece_start:piece_stop])
        output[output_start:output_stop] = np.take(source, indices, axis=0)
    return output


def _compact_source_reference(values: xr.DataArray):
    reference = getattr(values.data, _PBZ_SOURCE_ATTR, None)
    if not isinstance(reference, dict) or values.chunks is None:
        return None
    required = ("source", "array_path", "storage_options")
    if any(key not in reference for key in required):
        return None
    return {key: reference[key] for key in required}


def _open_source_array(reference):
    cache_key = tokenize(reference)
    with _OPEN_ARRAYS_LOCK:
        array = _OPEN_ARRAYS.get(cache_key)
        if array is None:
            kwargs = {}
            if reference["storage_options"] is not None:
                kwargs["storage_options"] = reference["storage_options"]
            array = zarr.open_array(
                reference["source"],
                mode="r",
                path=reference["array_path"],
                **kwargs,
            )
            _OPEN_ARRAYS[cache_key] = array
    return array


def _read_gather_block(
    reference,
    source_slices,
    pieces,
    output_shape,
) -> np.ndarray:
    array = _open_source_array(reference)
    blocks = tuple(
        np.asarray(array[tuple(slice(start, stop) for start, stop in bounds)])
        for bounds in source_slices
    )
    return _gather_block(blocks, pieces, output_shape)


def reduce_regions_dataset(
    ds: xr.Dataset,
    intervals,
    reducer: str,
    /,
    *,
    target_bytes: int = _TARGET_BYTES,
    max_source_blocks: int = _MAX_SOURCE_BLOCKS,
    **kwargs,
) -> xr.Dataset:
    """Reduce genomic intervals, reopening unchanged PBZ arrays in each task."""
    _validate_reducer(reducer, kwargs)
    packed = regions_dataset(
        ds,
        intervals,
        target_bytes=target_bytes,
        max_source_blocks=max_source_blocks,
    )
    return _reduce_packed_dataset(packed, reducer, **kwargs)


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
    plans = {}
    gather_batches = {}
    for name, values in ds.data_vars.items():
        plan_key = (
            values.shape,
            values.chunks,
            np.dtype(values.dtype).itemsize,
        )
        plan = plans.get(plan_key)
        if plan is None:
            plan = _plan_variable_regions(
                values,
                layout,
                target_bytes=target_bytes,
                max_source_blocks=max_source_blocks,
            )
            plans[plan_key] = plan
        batch_regions, batch_pieces, pieces = plan
        if values.chunks is None:
            gathered = _gather_eager(values, pieces, layout.total_positions)
        else:
            batches = gather_batches.get(plan_key)
            if batches is None:
                batches = _prepare_gather_batches(
                    values,
                    layout,
                    batch_regions,
                    batch_pieces,
                    pieces,
                )
                gather_batches[plan_key] = batches
            gathered = _gather_dask(
                values,
                layout,
                batch_regions,
                batch_pieces,
                pieces,
                batches=batches,
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


def _prepare_gather_batches(
    values: xr.DataArray,
    layout: RegionLayout,
    batch_region_edges: np.ndarray,
    batch_piece_edges: np.ndarray,
    pieces: np.ndarray,
):
    position_boundaries = np.concatenate(
        ([0], np.cumsum(values.chunks[0], dtype=np.int64))
    )
    column_boundaries = (
        np.concatenate(([0], np.cumsum(values.chunks[1], dtype=np.int64)))
        if values.ndim == 2
        else None
    )

    def source_slices(unique_source_blocks, column_block):
        slices = []
        for source_block in unique_source_blocks:
            source_block = int(source_block)
            bounds = [
                (
                    int(position_boundaries[source_block]),
                    int(position_boundaries[source_block + 1]),
                )
            ]
            if column_block is not None:
                bounds.append(
                    (
                        int(column_boundaries[column_block]),
                        int(column_boundaries[column_block + 1]),
                    )
                )
            slices.append(tuple(bounds))
        return tuple(slices)

    batches = []
    for batch_index in range(len(batch_region_edges) - 1):
        region_start = int(batch_region_edges[batch_index])
        region_stop = int(batch_region_edges[batch_index + 1])
        piece_start = int(batch_piece_edges[batch_index])
        piece_stop = int(batch_piece_edges[batch_index + 1])
        batch_view = pieces[piece_start:piece_stop]
        source_block_ids = batch_view["source_block"]
        source_block_starts = np.empty(len(batch_view), dtype=bool)
        source_block_starts[0] = True
        source_block_starts[1:] = source_block_ids[1:] != source_block_ids[:-1]
        unique_source_blocks = source_block_ids[source_block_starts]
        local_pieces = batch_view.copy()
        local_pieces["source_block"] = (
            np.cumsum(source_block_starts, dtype=local_pieces.dtype["source_block"])
            - 1
        )
        local_pieces.setflags(write=False)
        position_size = int(
            layout.packed_offsets[region_stop]
            - layout.packed_offsets[region_start]
        )
        source_slices_by_column = tuple(
            source_slices(unique_source_blocks, column_block)
            for column_block in (
                range(len(values.chunks[1])) if values.ndim == 2 else (None,)
            )
        )
        batches.append(
            (
                unique_source_blocks,
                local_pieces,
                position_size,
                source_slices_by_column,
            )
        )
    return tuple(batches)


def _gather_dask(
    values: xr.DataArray,
    layout: RegionLayout,
    batch_region_edges: np.ndarray,
    batch_piece_edges: np.ndarray,
    pieces: np.ndarray,
    *,
    batches=None,
):
    if batches is None:
        batches = _prepare_gather_batches(
            values,
            layout,
            batch_region_edges,
            batch_piece_edges,
            pieces,
        )
    del layout, batch_region_edges, batch_piece_edges, pieces
    reference = _compact_source_reference(values)
    source_blocks = None if reference is not None else values.data.to_delayed()
    rows = []
    for (
        unique_source_blocks,
        local_pieces,
        position_size,
        source_slices_by_column,
    ) in batches:
        if values.ndim == 1:
            if reference is None:
                selected = tuple(
                    source_blocks[(int(source_block),)]
                    for source_block in unique_source_blocks
                )
                task = delayed(_gather_block)(
                    selected,
                    local_pieces,
                    (position_size,),
                )
            else:
                task = delayed(_read_gather_block)(
                    reference,
                    source_slices_by_column[0],
                    local_pieces,
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
            if reference is None:
                selected = tuple(
                    source_blocks[(int(source_block), column_block)]
                    for source_block in unique_source_blocks
                )
                task = delayed(_gather_block)(
                    selected,
                    local_pieces,
                    (position_size, int(column_size)),
                )
            else:
                task = delayed(_read_gather_block)(
                    reference,
                    source_slices_by_column[column_block],
                    local_pieces,
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
    local_pieces["source_block"] = np.arange(
        len(local_pieces), dtype=local_pieces.dtype["source_block"]
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
