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


def build_peak_store(source, intervals, dest, *, tracks=None, chunk_size=None):
    """Convenience: build a region-mode store on disk at `dest`. Returns a PeakStore."""
    return RegionView(source, intervals, tracks=tracks, chunk_size=chunk_size).to_zarr(dest)


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
