"""POC: region-mode ("peak") stores — a lazy RegionView, compute/to_zarr, reduce.

A peak-store is a pbz store whose ragged index enumerates *regions* (peaks)
instead of contigs: `values` is sized Σ(peak lengths), `offsets` is the
per-region prefix-sum, and provenance rides on `region_contig/start/stop`
(replacing the `contigs` array). Marked with `perbase:segmentation="region"`.

Verbs mimic dask/xarray (lazy → concrete):

    rv = store.dataset(tracks).pbz.region_view(peaks)   # lazy gather, nothing decoded
    pv = rv.compute()                                   # build B in memory (MemoryStore)
    pv = rv.to_zarr("peaks.pbz")                        # build B on disk
    pv.mean().to_dataframe()                            # region × feature × sample table
    pv.top(by=["fire_coverage", "coverage"])            # winning row + genomic summit pos
    pv = pbzarr.PeakStore("peaks.pbz")                  # rediscover across sessions

Builder is a dead-simple per-region gather (bounded memory, slow at genome
scale). Reductions are pure-numpy `reduceat` over a region-aligned rechunk, so
each dask block holds whole regions and there is no boundary fold. numba and the
fast interval-native builder are deferred; this is correctness + ergonomics.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ._reduce import _normalize_intervals, compute_boundaries

_MARKER = [{"uuid": "perbase-region", "name": "perbase"}]
_DEFAULT_CHUNK = 1_000_000


# ------------------------------------------------------------- RegionView ---

class RegionView:
    """A lazy description of the in-peak gather. `compute()` → memory; `to_zarr()` → disk."""

    def __init__(self, source, intervals, *, tracks=None, chunk_size=None):
        self._source = source
        self._intervals = intervals
        self._tracks = tracks
        self._chunk_size = chunk_size

    def compute(self):
        """Build B in an in-memory zarr MemoryStore. Returns a PeakStore."""
        import zarr

        root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
        _build_into(root, self._source, self._intervals, self._tracks, self._chunk_size)
        return PeakStore(root)

    def to_zarr(self, path):
        """Build B on disk at `path`. Returns a PeakStore."""
        import zarr

        root = zarr.open_group(str(path), mode="w")
        _build_into(root, self._source, self._intervals, self._tracks, self._chunk_size)
        return PeakStore(root)

    def to_pbz(self, path, *, source=None, shard_size=None, column_chunk_size=None,
               workers=None, decode_workers=None, write_workers=None,
               writer_queue_depth=2, progress=False):
        """Build B on disk at `path` with the Rust region-store builder.

        The fast, single-process, all-cores path (no dask cluster). Requires the
        region-view's source to be an on-disk pbz store: `source` is that store's
        path, defaulting to the one the source dataset was opened from
        (`ds.pbz.region_view(...)` stamps it). Returns a `PeakStore(path)`.
        """
        from ._native import build_region_store

        src_path = source
        if src_path is None:
            src_path = getattr(self._source, "attrs", {}).get("pbz_source_path")
        if src_path is None:
            raise ValueError(
                "to_pbz needs an on-disk source store; pass source=<path> or build the "
                "region view from a store dataset (store.dataset(...).pbz.region_view(...))"
            )

        contigs, starts, ends = _interval_arrays(self._intervals)
        tracks = list(self._tracks) if self._tracks is not None else None
        build_region_store(
            str(src_path), contigs, starts, ends, str(path),
            tracks=tracks, chunk_size=self._chunk_size, shard_size=shard_size,
            column_chunk_size=column_chunk_size, workers=workers,
            decode_workers=decode_workers, write_workers=write_workers,
            writer_queue_depth=writer_queue_depth, progress=progress,
        )
        return PeakStore(str(path))


def _interval_arrays(intervals):
    """(contig_names, starts, ends) as plain Python lists for the Rust builder."""
    from ._reduce import _columnar_intervals
    from ._region import RegionQuery

    columnar = _columnar_intervals(intervals)
    if columnar is not None:
        names, starts, ends = columnar
        return (
            [str(x) for x in np.asarray(names)],
            [int(x) for x in np.asarray(starts)],
            [int(x) for x in np.asarray(ends)],
        )
    names, starts, ends = [], [], []
    for item in intervals:
        if isinstance(item, RegionQuery):
            n, s, e = item.contig, item.start, item.end
        else:
            n, s, e = item[0], item[1], item[2]
        if s is None or e is None:
            raise ValueError(f"interval {item!r} needs explicit start and end")
        names.append(str(n))
        starts.append(int(s))
        ends.append(int(e))
    return names, starts, ends


def build_peak_store(source, intervals, dest, *, tracks=None, chunk_size=None):
    """Convenience: build a region-mode store on disk at `dest`. Returns a PeakStore."""
    return RegionView(source, intervals, tracks=tracks, chunk_size=chunk_size).to_zarr(dest)


def build_peak_store_parallel(store, intervals, dest, *, tracks=None, chunk_size=None,
                              shard_size=None, read_window=None, client=None):
    """Per-OUTPUT-WRITE-UNIT builder: write-unit-aligned, race-free, parallel over processes.

    `store` is a `PbzStore` or store path (source must be an on-disk pbz store). One task per
    (track, write-unit): each **streams** its regions' source span in chunk-aligned windows,
    scatters the in-peak rows into a full-write-unit buffer, and writes it in one shot. Array
    handles are memoized per worker process (`_open`), so a long-lived worker opens each array
    once and reuses it across every task — no per-task reopen storm.

    `chunk_size` is the inner chunk (compression/read granularity). `shard_size`, if given,
    turns on sharding: many inner chunks bundle into one shard file (fewer files → less
    Lustre metadata storm), and the **write unit becomes the shard** (whole-shard writes are
    the only race-free / single-encode path); must be a multiple of `chunk_size`. `read_window`
    is the source read granularity in rows (default: 8 × the source's on-disk chunk); larger
    means fewer/larger reads and more concurrent chunk decode per read (good on Lustre) at the
    cost of `read_window × n_cols × dtype` more memory per task — decoupled from both the
    on-disk chunk and the write unit. Peak per-task memory ≈ write-unit buffer + one window.
    With `client` tasks run across processes; without, serially in-process. Returns a
    `PeakStore`.
    """
    import zarr

    from ._pbzstore import PbzStore
    from ._read import open_track

    store_path = store.path if isinstance(store, PbzStore) else str(store)
    names = PbzStore(store_path).tracks() if tracks is None else list(tracks)

    first = open_track(store_path, names[0])
    contigs = list(first.contigs)
    offsets = np.asarray(first.offsets)
    contig_ids, starts, ends = _normalize_intervals(intervals, contigs)
    sorted_starts, sorted_ends, order = compute_boundaries(contig_ids, starts, ends, offsets)
    k = len(sorted_starts)
    if k == 0:
        raise ValueError("build: no intervals")
    lengths = (sorted_ends - sorted_starts).astype(np.int64)
    region_offsets = np.empty(k + 1, dtype=np.int64)
    region_offsets[0] = 0
    np.cumsum(lengths, out=region_offsets[1:])
    total = int(region_offsets[-1])
    inner = int(chunk_size or min(_DEFAULT_CHUNK, total))
    if shard_size is not None:
        wu = int(shard_size)
        if wu % inner != 0:
            raise ValueError(f"shard_size {wu} must be a multiple of chunk_size {inner}")
    else:
        wu = inner  # write unit == inner chunk when not sharding
    cw = wu         # tasks step by the write unit

    src_contig = np.array([contigs[c] for c in contig_ids[order]], dtype=object)
    src_start = np.asarray(starts, dtype=np.int64)[order]
    src_stop = np.asarray(ends, dtype=np.int64)[order]
    input_index = np.asarray(order, dtype=np.int64)

    # create B structure + aux on the client (tasks only write `values` chunks)
    root = zarr.open_group(str(dest), mode="w")
    root.attrs["zarr_conventions"] = _MARKER
    root.attrs["perbase:kind"] = "collection"
    root.attrs["perbase:segmentation"] = "region"
    tasks = []
    for name in names:
        ta = open_track(store_path, name)
        is_2d = ta.col_dim is not None
        ncol = len(ta.labels) if is_2d else None
        gmeta = dict(zarr.open_group(f"{store_path}/{name}", mode="r").attrs)

        g = root.create_group(name)
        g.attrs["zarr_conventions"] = _MARKER
        g.attrs["perbase:kind"] = "track"
        g.attrs["perbase:version"] = "0.4"
        g.attrs["perbase:segmentation"] = "region"
        g.attrs["perbase:coordinates"] = "0-based-half-open"
        gc = gmeta.get("perbase:genome_checksum")
        if gc:
            g.attrs["perbase:parent_genome_checksum"] = gc
        if is_2d:
            shape, chunks, dims = (total, ncol), (inner, ncol), ["position", ta.col_dim]
            shards = (wu, ncol) if shard_size is not None else None
        else:
            shape, chunks, dims = (total,), (inner,), ["position"]
            shards = (wu,) if shard_size is not None else None
        create_kw = {} if shards is None else {"shards": shards}
        g.create_array("values", shape=shape, chunks=chunks, dtype=ta.values.dtype,
                       dimension_names=dims, **create_kw)
        _aux(g, "offsets", region_offsets, "region_boundary")
        _aux(g, "region_contig", src_contig, "region")
        _aux(g, "region_start", src_start, "region")
        _aux(g, "region_stop", src_stop, "region")
        _aux(g, "region_input_index", input_index, "region")
        if is_2d:
            _aux(g, ta.col_dim, np.array(ta.labels, dtype=object), ta.col_dim)
            g.attrs["perbase:column_dim"] = ta.col_dim

        src_path = f"{store_path}/{name}/values"
        dst_path = f"{dest}/{name}/values"
        src_inner = int(ta.values.chunks[0])
        window_rows = int(read_window) if read_window is not None else 8 * src_inner
        window_rows = max(window_rows, src_inner)
        n_chunks = (total + cw - 1) // cw
        for c in range(n_chunks):
            o0, o1 = c * cw, min((c + 1) * cw, total)
            r0 = int(np.searchsorted(region_offsets, o0, side="right") - 1)
            r1 = int(np.searchsorted(region_offsets, o1 - 1, side="right") - 1)
            off_slice = region_offsets[r0:r1 + 2].copy()
            ss_slice = sorted_starts[r0:r1 + 1].copy()
            tasks.append((src_path, dst_path, is_2d, ncol, o0, o1, off_slice, ss_slice,
                          window_rows))

    _clear_open_cache()  # drop any stale handles from a prior build at a reused path
    if client is not None:
        client.run(_clear_open_cache)
        futures = [client.submit(_write_output_chunk, *t, pure=False) for t in tasks]
        client.gather(futures)
    else:
        for t in tasks:
            _write_output_chunk(*t)
    return PeakStore(str(dest))


@functools.lru_cache(maxsize=None)
def _open(path, mode):
    """Memoized array handle. Worker processes are long-lived, so each opens a given
    array once and reuses it across every task on that worker — no per-task reopen storm.

    Cleared at the start of each build (`_clear_open_cache`), since a path may be recreated
    across builds in one process and a cached handle would then point at stale/deleted bytes.
    """
    import zarr

    return zarr.open_array(path, mode=mode)


def _clear_open_cache():
    _open.cache_clear()


def _write_output_chunk(src_path, dst_path, is_2d, ncol, o0, o1, off_slice, ss_slice,
                        window_rows):
    """Assemble one whole output write-unit by streaming the source and scatter-writing it.

    Output rows are gap-free (regions tile the axis), so the buffer is fully covered. The
    source is read in chunk-aligned windows of `window_rows`; each window is decoded once,
    the in-peak rows it covers are scattered into the buffer, then it is dropped — so peak
    memory is one window plus the output buffer, not the whole min..max slab. Pure-gap
    windows (no overlapping region) are skipped, capping read amplification.
    """
    import numpy as np

    src = _open(src_path, "r")
    dst = _open(dst_path, "r+")
    n = o1 - o0
    segs = []  # (out_lo, out_hi, src_lo, src_hi)
    for j in range(len(ss_slice)):
        a = max(o0, int(off_slice[j]))
        b = min(o1, int(off_slice[j + 1]))
        if b <= a:
            continue
        s = int(ss_slice[j])
        segs.append((a - o0, b - o0, s + (a - int(off_slice[j])), s + (b - int(off_slice[j]))))

    buf = np.empty((n, ncol) if is_2d else (n,), dtype=src.dtype)
    lo = min(s[2] for s in segs)
    hi = max(s[3] for s in segs)
    for w0 in range(lo - (lo % window_rows), hi, window_rows):
        w1 = min(w0 + window_rows, hi)
        if not any(sl < w1 and w0 < sh for _, _, sl, sh in segs):
            continue  # pure-gap window: nothing to gather
        window = np.asarray(src[w0:w1, :]) if is_2d else np.asarray(src[w0:w1])
        for ol, _oh, sl, sh in segs:
            a = max(sl, w0)
            b = min(sh, w1)
            if b <= a:
                continue
            oa, ob = ol + (a - sl), ol + (b - sl)
            if is_2d:
                buf[oa:ob, :] = window[a - w0:b - w0, :]
            else:
                buf[oa:ob] = window[a - w0:b - w0]

    if is_2d:
        dst[o0:o1, :] = buf
    else:
        dst[o0:o1] = buf


# ---------------------------------------------------------------- build (B) ---

def _build_into(root, source, intervals, tracks, chunk_size):
    contigs = [str(c) for c in np.asarray(source["contigs"].values)]
    offsets = np.asarray(source["offsets"].values)
    contig_ids, starts, ends = _normalize_intervals(intervals, contigs)
    sorted_starts, sorted_ends, order = compute_boundaries(contig_ids, starts, ends, offsets)
    k = len(sorted_starts)
    if k == 0:
        raise ValueError("build: no intervals")

    lengths = (sorted_ends - sorted_starts).astype(np.int64)
    region_offsets = np.empty(k + 1, dtype=np.int64)
    region_offsets[0] = 0
    np.cumsum(lengths, out=region_offsets[1:])
    total = int(region_offsets[-1])

    names = list(source.data_vars) if tracks is None else list(tracks)
    cw = int(chunk_size or min(_DEFAULT_CHUNK, total))

    root.attrs["zarr_conventions"] = _MARKER
    root.attrs["perbase:kind"] = "collection"
    root.attrs["perbase:segmentation"] = "region"

    # provenance, in sorted flat order (matching the values layout). region_input_index
    # keeps the map back to the caller's original interval order for downstream joins.
    src_contig = np.array([contigs[c] for c in contig_ids[order]], dtype=object)
    src_start = np.asarray(starts, dtype=np.int64)[order]
    src_stop = np.asarray(ends, dtype=np.int64)[order]
    input_index = np.asarray(order, dtype=np.int64)

    gc = source.attrs.get("genome_checksum")
    for name in names:
        da = source[name]
        col_dim = _col_dim(da)
        labels = [str(x) for x in np.asarray(da[col_dim].values)] if col_dim else None
        g = root.create_group(name)
        g.attrs["zarr_conventions"] = _MARKER
        g.attrs["perbase:kind"] = "track"
        g.attrs["perbase:version"] = "0.4"
        g.attrs["perbase:segmentation"] = "region"
        g.attrs["perbase:coordinates"] = "0-based-half-open"
        if gc:
            g.attrs["perbase:parent_genome_checksum"] = gc

        if col_dim:
            shape, chunks, dims = (total, len(labels)), (cw, len(labels)), ["position", col_dim]
        else:
            shape, chunks, dims = (total,), (cw,), ["position"]
        vals = g.create_array(
            "values", shape=shape, chunks=chunks, dtype=da.dtype, dimension_names=dims
        )
        for i in range(k):
            lo, hi = int(sorted_starts[i]), int(sorted_ends[i])
            out_lo, out_hi = int(region_offsets[i]), int(region_offsets[i + 1])
            vals[out_lo:out_hi] = np.asarray(da.isel(position=slice(lo, hi)).values)

        _aux(g, "offsets", region_offsets, "region_boundary")
        _aux(g, "region_contig", src_contig, "region")
        _aux(g, "region_start", src_start, "region")
        _aux(g, "region_stop", src_stop, "region")
        _aux(g, "region_input_index", input_index, "region")
        if col_dim:
            _aux(g, col_dim, np.array(labels, dtype=object), col_dim)
            g.attrs["perbase:column_dim"] = col_dim
    return root


def _aux(group, name, data, dim):
    data = np.asarray(data)
    dtype = str if data.dtype == object else data.dtype
    arr = group.create_array(
        name, shape=data.shape, chunks=data.shape or (1,), dtype=dtype, dimension_names=[dim]
    )
    arr[...] = data


def _col_dim(da):
    extra = [d for d in da.dims if d != "position"]
    return extra[0] if extra else None


# ------------------------------------------------------------- read (B) ---

@dataclass
class _Feature:
    name: str
    values: "object"
    col_dim: str | None
    labels: list[str] | None


class PeakStore:
    """A region-mode store: reductions run over the CSR region segmentation.

    Constructed from a path (`PeakStore("peaks.pbz")`) or an open zarr Group
    (what `RegionView.compute()` hands back, MemoryStore-backed).
    """

    def __init__(self, path_or_group):
        import zarr

        from ._kind import is_perbase
        from ._native import PbzError

        if isinstance(path_or_group, (str, bytes)):
            self.root = zarr.open_group(str(path_or_group), mode="r")
        else:
            self.root = path_or_group
        attrs = dict(self.root.attrs)
        if not is_perbase(attrs):
            raise PbzError("not a pbz store")
        if attrs.get("perbase:segmentation") != "region":
            raise PbzError("not a region-mode (peak) store")

        self._names = sorted(
            n for n, node in self.root.members()
            if dict(node.attrs).get("perbase:segmentation") == "region"
        )
        g0 = self.root[self._names[0]]
        self.offsets = np.asarray(g0["offsets"][:])
        self.region_contig = [str(c) for c in g0["region_contig"][:]]
        self.region_start = np.asarray(g0["region_start"][:])
        self.region_stop = np.asarray(g0["region_stop"][:])
        self.region_input_index = np.asarray(g0["region_input_index"][:])
        self.n_regions = len(self.offsets) - 1

    def features(self) -> list[str]:
        return list(self._names)

    def _feature(self, name: str) -> _Feature:
        vals = self.root[name]["values"]
        dims = tuple(vals.metadata.dimension_names)
        col_dim = dims[1] if len(dims) == 2 else None
        labels = [str(x) for x in self.root[name][col_dim][:]] if col_dim else None
        return _Feature(name, vals, col_dim, labels)

    def _block_bounds(self, target=_DEFAULT_CHUNK):
        offs = self.offsets
        bounds = [0]
        cur = 0
        for i in range(self.n_regions):
            if offs[i + 1] - offs[cur] >= target and offs[i + 1] > offs[cur]:
                bounds.append(int(offs[i + 1]))
                cur = i + 1
        if bounds[-1] != int(offs[-1]):
            bounds.append(int(offs[-1]))
        return np.asarray(bounds)

    def _dask_values(self, feat: _Feature):
        import dask.array as da

        arr = da.from_array(feat.values, chunks=feat.values.chunks)
        bounds = self._block_bounds()
        sizes = tuple(int(x) for x in np.diff(bounds))
        rechunk = (sizes, -1) if feat.col_dim else (sizes,)
        return arr.rechunk(rechunk), bounds

    def reduce(self, func: str):
        import xarray as xr

        data_vars = {n: self._reduce_one(self._feature(n), func) for n in self._names}
        return xr.Dataset(data_vars).assign_coords(self._region_coords())

    def _reduce_one(self, feat: _Feature, func: str):
        import dask.array as da
        import xarray as xr

        vals, bounds = self._dask_values(feat)
        starts_by_block = _starts_per_block(self.offsets, bounds)
        op = _REDUCEAT_OP[func]
        ncol = len(feat.labels) if feat.col_dim else None

        def block(a, block_info=None):
            bi = block_info[0]["array-location"][0][0]
            return op(a, starts_by_block[bounds.searchsorted(bi)], feat)

        out_chunks = (tuple(len(s) for s in starts_by_block[:-1]),)
        if ncol:
            out_chunks = out_chunks + ((ncol,),)
        dtype = _out_dtype(feat.values.dtype, func)
        meta = np.empty((0, ncol) if ncol else (0,), dtype=dtype)
        reduced = da.map_blocks(block, vals, chunks=out_chunks, dtype=dtype, meta=meta)
        dims = ("region", feat.col_dim) if feat.col_dim else ("region",)
        return xr.DataArray(reduced, dims=dims)

    def mean(self):
        return self.reduce("mean")

    def sum(self):
        return self.reduce("sum")

    def min(self):
        return self.reduce("min")

    def max(self):
        return self.reduce("max")

    def count(self):
        return self.reduce("count")

    def top(self, by, *, descending=True):
        """Winning row per region ordered by `by`; attaches the genomic summit position.

        Returns every feature at the winning position per region (per column),
        tie-broken toward the earliest position, plus `summit_pos` (genomic) and
        `summit_contig` coords. Lazy build of the winner indices; small gather.
        """
        import dask.array as da
        import xarray as xr

        keys = by if isinstance(by, (list, tuple)) else [by]
        descs = list(descending) if isinstance(descending, (list, tuple)) else [descending] * len(keys)

        key_feats = [self._feature(k) for k in keys]
        col_dim = key_feats[0].col_dim
        ncol = len(key_feats[0].labels) if col_dim else None
        packed_vals = [self._dask_values(f)[0] for f in key_feats]
        bounds = self._block_bounds()
        starts_by_block = _starts_per_block(self.offsets, bounds)

        def winners_block(*arrs, block_info=None):
            bi = block_info[0]["array-location"][0][0]
            return _segment_argmax(arrs, descs, starts_by_block[bounds.searchsorted(bi)], ncol)

        win_chunks = (tuple(len(s) for s in starts_by_block[:-1]),)
        if ncol:
            win_chunks = win_chunks + ((ncol,),)
        winners = np.asarray(da.map_blocks(
            winners_block, *packed_vals, chunks=win_chunks, dtype=np.int64,
            meta=np.empty((0, ncol) if ncol else (0,), dtype=np.int64),
        ))  # (region[, col]) absolute row index into the flat values axis

        data_vars = {n: self._gather_at(self._feature(n), winners, ncol) for n in self._names}
        ds = xr.Dataset(data_vars).assign_coords(self._region_coords())
        return ds.assign_coords(self._summit_coords(winners, col_dim, ncol))

    def _gather_at(self, feat: _Feature, winners, ncol):
        import xarray as xr

        vals = feat.values
        if feat.col_dim:
            cols = np.arange(ncol)
            gathered = np.asarray(vals[:])[winners, cols[None, :]]
            dims = ("region", feat.col_dim)
        else:
            w = winners if winners.ndim == 1 else winners[:, 0]
            gathered = np.asarray(vals[:])[w]
            dims = ("region",)
        return xr.DataArray(gathered, dims=dims)

    def _summit_coords(self, winners, col_dim, ncol):
        # genomic position of the winning row: region_start[r] + (row - region_offsets[r])
        roff = self.offsets[:-1]
        rstart = self.region_start
        if col_dim:
            gpos = rstart[:, None] + (winners - roff[:, None])
            summit = ("region", col_dim), gpos
        else:
            w = winners if winners.ndim == 1 else winners[:, 0]
            summit = ("region",), rstart + (w - roff)
        return {
            "summit_pos": summit,
            "summit_contig": ("region", np.array(self.region_contig)),
        }

    def _region_coords(self):
        return {
            "region": np.arange(self.n_regions),
            "region_contig": ("region", np.array(self.region_contig)),
            "region_start": ("region", self.region_start),
            "region_stop": ("region", self.region_stop),
            "region_input_index": ("region", self.region_input_index),
        }

    def dataset(self):
        import dask.array as da
        import xarray as xr

        data_vars = {}
        for name in self._names:
            feat = self._feature(name)
            arr = da.from_array(feat.values, chunks=feat.values.chunks)
            if feat.col_dim:
                data_vars[name] = xr.DataArray(
                    arr, dims=["position", feat.col_dim],
                    coords={feat.col_dim: np.asarray(feat.labels)},
                )
            else:
                data_vars[name] = xr.DataArray(arr, dims=["position"])
        return xr.Dataset(data_vars)


# ------------------------------------------------------- kernels (numpy) ---

def _starts_per_block(offs, bounds):
    per = []
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        per.append((offs[(offs >= lo) & (offs < hi)] - lo).astype(np.int64))
    per.append(np.asarray([], dtype=np.int64))
    return per


_REDUCEAT_OP = {}


def _register(name, ufunc):
    def op(a, loc, feat):
        if loc.size == 0:
            width = a.shape[1:] if a.ndim > 1 else ()
            return np.empty((0, *width), dtype=a.dtype)
        return ufunc.reduceat(a, loc, axis=0)
    _REDUCEAT_OP[name] = op


_register("sum", np.add)
_register("min", np.minimum)
_register("max", np.maximum)


def _mean_op(a, loc, feat):
    if loc.size == 0:
        width = a.shape[1:] if a.ndim > 1 else ()
        return np.empty((0, *width), dtype=np.float64)
    s = np.add.reduceat(a.astype(np.float64), loc, axis=0)
    lengths = np.diff(np.append(loc, a.shape[0])).astype(np.float64)
    if a.ndim > 1:
        lengths = lengths[:, None]
    return s / lengths


def _count_op(a, loc, feat):
    if loc.size == 0:
        return np.empty((0,), dtype=np.int64)
    lengths = np.diff(np.append(loc, a.shape[0])).astype(np.int64)
    if a.ndim > 1:
        lengths = np.repeat(lengths[:, None], a.shape[1], axis=1)
    return lengths


_REDUCEAT_OP["mean"] = _mean_op
_REDUCEAT_OP["count"] = _count_op


def _out_dtype(in_dtype, func):
    if func == "mean":
        return np.float64
    if func == "count":
        return np.int64
    return np.dtype(in_dtype)


def _segment_argmax(arrs, descs, loc, ncol):
    """Per-region winning GLOBAL row index by lexicographic (arrs, descs).

    Tie-break toward the earliest position. Pure numpy over reduceat. NOTE: POC
    packs keys as float64 — not bit-exact for very large counts; replace with the
    packed-int scheme / numba kernel when hardening.
    """
    a0 = arrs[0]
    n = a0.shape[0]
    if loc.size == 0:
        return np.empty((0, ncol) if ncol else (0,), dtype=np.int64)
    lengths = np.diff(np.append(loc, n))
    key = np.zeros(a0.shape, dtype=np.float64)
    scale = 1.0
    for a, desc in zip(reversed(arrs), reversed(descs)):
        v = a.astype(np.float64)
        span = float(v.max()) + 1.0 if v.size else 1.0
        key = key + (v if desc else (span - 1.0 - v)) * scale
        scale *= span
    seg_max = np.maximum.reduceat(key, loc, axis=0)
    is_win = key == np.repeat(seg_max, lengths, axis=0)
    pos = np.arange(n)
    if key.ndim > 1:
        pos = np.repeat(pos[:, None], key.shape[1], axis=1)
    masked = np.where(is_win, pos, np.iinfo(np.int64).max)
    return np.minimum.reduceat(masked, loc, axis=0).astype(np.int64)
