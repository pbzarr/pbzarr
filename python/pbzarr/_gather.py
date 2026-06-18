"""Region gather core.

A pure planner (chunk math, zero I/O) plus eager-threadpool and lazy-dask
executors shared by `PbzStore.region` / `region_reduced` / `region_blocks`.

The planner partitions each region into per-block contiguous segments at
inner-chunk granularity, so zarr range-GETs only the chunks a region touches and
each touched block is read exactly once. The executors differ only in how they
run the reads; the chunk math is written once.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr


@dataclass
class RegionBlocks:
    """Raw per-region values from `region_blocks`, aligned to input order.

    `blocks[i]` holds region `i`'s values: `(n_positions, n_columns)` for a 2D
    track, `(n_positions,)` for a scalar track. `columns` is the shared column
    label array (`None` for scalar tracks), returned once rather than per region.
    `regions[i]` is the resolved `(contig, start, end)` for block `i`.
    """

    blocks: list[np.ndarray]
    columns: np.ndarray | None
    regions: list[tuple[str, int, int]]


# A planned segment: a contiguous run of positions inside one block, tagged with
# the global input index of the region it belongs to. Block-relative so the
# executor slices it directly (`block[rel_start:rel_end]`) with no fancy index.
def plan_blocks(
    regions: list[tuple[int, int, int]], n_positions: int, block_size: int
) -> dict[int, list[tuple[int, int, int]]]:
    """Partition regions into per-block segments. Pure: no I/O.

    `regions` is a list of `(region_id, start, end)` for one contig, with `start`
    / `end` already resolved (0-based half-open, clipped to `[0, n_positions]`).
    Returns `{block_id: [(rel_start, rel_end, region_id), ...]}`. A region that
    crosses a block boundary contributes one segment per block it touches, all
    under the same `region_id`.
    """
    by_block: dict[int, list[tuple[int, int, int]]] = {}
    for region_id, start, end in regions:
        if end <= start:
            continue
        for b in range(start // block_size, (end - 1) // block_size + 1):
            lo = b * block_size
            hi = min(lo + block_size, n_positions)
            by_block.setdefault(b, []).append(
                (max(start, lo) - lo, min(end, hi) - lo, region_id)
            )
    return by_block


def resolve_region(start: int | None, end: int | None, n: int) -> tuple[int, int]:
    """Clip a (possibly open-ended) region to `[0, n]`, 0-based half-open."""
    s = 0 if start is None else max(start, 0)
    e = n if end is None else min(end, n)
    return s, max(s, e)


def default_workers(path: str, workers: int | None) -> int:
    """Thread count for the eager executor.

    Local reads are CPU-bound (decompress); remote reads are latency-bound and
    want many concurrent in-flight requests to hide round-trip latency. Caller
    override always wins.
    """
    if workers is not None:
        return max(1, workers)
    if "://" in path:
        return 64
    return min(32, (os.cpu_count() or 1) + 4)


# A read task: one touched block. `entries` are the segments to slice out of it,
# each carrying its position in the global ordering (`seq`) so results reassemble
# deterministically regardless of completion order.
_Task = tuple[zarr.Array, int, int, list[tuple[int, int, int, int]]]


def _plan_tasks(
    group: zarr.Group,
    track: str,
    rqs_by_contig: dict[str, list[tuple[int, int, int]]],
    store_contigs: list[str],
) -> tuple[list[_Task], int, bool, int, Any]:
    """Build the ordered read-task list across all touched contigs.

    Returns `(tasks, n_segments, is_2d, ncol, dtype)`. Segments are numbered in
    contig-then-ascending-block order; within a region that order is also
    position order, so block reassembly is a stable group-by.
    """
    tasks: list[_Task] = []
    seq = 0
    is_2d = False
    ncol = 1
    dtype: Any = np.float32
    for contig in store_contigs:
        if contig not in rqs_by_contig:
            continue
        arr = _array(_group(group, contig), track)
        dtype = arr.dtype
        is_2d = arr.ndim == 2
        ncol = int(arr.shape[1]) if is_2d else 1
        n = int(arr.shape[0])
        block_size = int(arr.chunks[0])
        by_block = plan_blocks(rqs_by_contig[contig], n, block_size)
        for b in sorted(by_block):
            lo = b * block_size
            segs = by_block[b]
            hi_rel = max(re for _, re, _ in segs)
            entries = [(seq + i, rs, re, rid) for i, (rs, re, rid) in enumerate(segs)]
            seq += len(segs)
            tasks.append((arr, lo, hi_rel, entries))
    return tasks, seq, is_2d, ncol, dtype


def _read_task(task: _Task, is_2d: bool) -> list[tuple[int, int, np.ndarray]]:
    arr, lo, hi_rel, entries = task
    block = np.asarray(arr[lo : lo + hi_rel])
    out = []
    for seq, rs, re, rid in entries:
        v = block[rs:re, :] if is_2d else block[rs:re]
        out.append((seq, rid, v))
    return out


def _eager_segments(
    tasks: list[_Task], n_seg: int, is_2d: bool, workers: int
) -> list[tuple[int, np.ndarray]]:
    """Read every touched block once in a thread pool; return per-segment values
    ordered by `seq` as `(region_id, values)`."""
    slots: list[tuple[int, np.ndarray] | None] = [None] * n_seg
    if tasks:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda t: _read_task(t, is_2d), tasks):
                for seq, rid, v in result:
                    slots[seq] = (rid, v)
    return [s for s in slots if s is not None]


def gather_tagged(
    group: zarr.Group,
    track: str,
    rqs_by_contig: dict[str, list[tuple[int, int, int]]],
    store_contigs: list[str],
    *,
    lazy: bool,
    workers: int,
) -> tuple[Any, np.ndarray, bool, int, Any]:
    """Gather many regions into one array along `position` plus a parallel
    `region_id` array. Eager → numpy; lazy → dask. Used by `region` (multi) and
    `region_reduced`."""
    tasks, n_seg, is_2d, ncol, dtype = _plan_tasks(
        group, track, rqs_by_contig, store_contigs
    )
    empty_shape = (0, ncol) if is_2d else (0,)
    if n_seg == 0:
        return np.empty(empty_shape, dtype=dtype), np.empty(0, np.int64), is_2d, ncol, dtype

    if lazy:
        import dask.array as dka

        sources: dict[int, Any] = {}
        pieces: list[Any] = [None] * n_seg
        rids: list[np.ndarray] = [None] * n_seg  # type: ignore[list-item]
        for arr, lo, hi_rel, entries in tasks:
            dz = sources.get(id(arr))
            if dz is None:
                dz = dka.from_zarr(arr)
                sources[id(arr)] = dz
            block_da = dz[lo : lo + hi_rel]
            for seq, rs, re, rid in entries:
                pieces[seq] = block_da[rs:re]
                rids[seq] = np.full(re - rs, rid, dtype=np.int64)
        gathered = dka.concatenate(pieces, axis=0)
        return gathered, np.concatenate(rids), is_2d, ncol, dtype

    segs = _eager_segments(tasks, n_seg, is_2d, workers)
    gathered = np.concatenate([v for _, v in segs], axis=0)
    region_ids = np.concatenate([np.full(len(v), rid, np.int64) for rid, v in segs])
    return gathered, region_ids, is_2d, ncol, dtype


def gather_blocks(
    group: zarr.Group,
    track: str,
    resolved: list[tuple[str, int, int]],
    rqs_by_contig: dict[str, list[tuple[int, int, int]]],
    store_contigs: list[str],
    *,
    workers: int,
) -> tuple[list[np.ndarray], bool, int, Any]:
    """Gather many regions as per-region numpy blocks aligned to input order.
    Eager only. `resolved[i]` is region `i`'s `(contig, start, end)`."""
    tasks, n_seg, is_2d, ncol, dtype = _plan_tasks(
        group, track, rqs_by_contig, store_contigs
    )
    segs = _eager_segments(tasks, n_seg, is_2d, workers)

    parts: dict[int, list[np.ndarray]] = {}
    for rid, v in segs:
        parts.setdefault(rid, []).append(v)

    empty_shape = (0, ncol) if is_2d else (0,)
    blocks: list[np.ndarray] = []
    for i in range(len(resolved)):
        pieces = parts.get(i)
        if pieces:
            blocks.append(np.concatenate(pieces, axis=0))
        else:
            blocks.append(np.empty(empty_shape, dtype=dtype))
    return blocks, is_2d, ncol, dtype


def _array(g: zarr.Group, name: str) -> zarr.Array:
    node = g[name]
    if not isinstance(node, zarr.Array):
        raise TypeError(f"expected {name!r} to be a zarr Array, got {type(node).__name__}")
    return node


def _group(g: zarr.Group, name: str) -> zarr.Group:
    node = g[name]
    if not isinstance(node, zarr.Group):
        raise TypeError(f"expected {name!r} to be a zarr Group, got {type(node).__name__}")
    return node
