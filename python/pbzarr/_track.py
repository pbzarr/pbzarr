"""Track: the standalone read/query unit over one flat track.

Holds (store_path, name). Materializes the track via import_*, then reads it
(metadata + region API) by resolving contig regions to flat slices. Mirrors the
Rust `Track`.
"""
from __future__ import annotations

import zarr

from . import _read
from ._read import RegionBlocks
from ._region import RegionQuery, parse_region


def _rq_list(query) -> list[RegionQuery]:
    if isinstance(query, str):
        return [parse_region(query)]
    if isinstance(query, tuple):
        return [RegionQuery(*query)]
    return [parse_region(q) if isinstance(q, str) else RegionQuery(*q) for q in query]


def _is_single(query) -> bool:
    return isinstance(query, (str, tuple))


def _norm_sources(sources) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for s in sources:
        if isinstance(s, (tuple, list)):
            out.append((str(s[0]), s[1] if len(s) > 1 else None))
        else:
            out.append((str(s), None))
    return out


class Track:
    def __init__(self, store_path: str, name: str):
        self.store_path = str(store_path)
        self.name = name

    def _values(self) -> "zarr.Array":
        return zarr.open_array(f"{self.store_path}/{self.name}/values", mode="r")

    @property
    def dtype(self) -> str:
        return str(self._values().dtype)

    @property
    def rank(self) -> int:
        return len(self._values().metadata.dimension_names)

    @property
    def column_dim(self) -> str | None:
        dims = self._values().metadata.dimension_names
        return dims[1] if len(dims) == 2 else None

    def total_len(self) -> int:
        return int(self._values().shape[0])

    def genome(self) -> list[tuple[str, int]]:
        ta = _read.open_track(self.store_path, self.name)
        return [(c, int(ta.offsets[i + 1] - ta.offsets[i])) for i, c in enumerate(ta.contigs)]

    def column_labels(self) -> list[str] | None:
        return _read.open_track(self.store_path, self.name).labels

    def region(self, query, *, column: str | None = None):
        rqs = _rq_list(query)
        if _is_single(query):
            return _read.read_region(self.store_path, self.name, rqs[0], column)
        return _read.gather_regions(self.store_path, self.name, rqs, column)

    def region_reduced(self, query, *, reduce: str, column: str | None = None):
        da = _read.gather_regions(self.store_path, self.name, _rq_list(query), column)
        grouped = da.groupby("region")
        try:
            reducer = getattr(grouped, reduce)
        except AttributeError as e:
            raise ValueError(f"unsupported reduce {reduce!r}") from e
        return reducer()

    def region_blocks(self, query, *, column: str | None = None) -> RegionBlocks:
        return _read.region_blocks(self.store_path, self.name, _rq_list(query), column)

    def dataset(self):
        import xarray as xr

        return xr.open_datatree(self.store_path, engine="zarr", consolidated=False)[self.name].to_dataset()

    def import_bed(self, sources, *, column: str, dtype: str, genome: str,
                   workers=None, chunk_size=None, column_chunk_size=None, progress=False) -> None:
        from ._native import import_bed

        import_bed(self.store_path, self.name, _norm_sources(sources), column, dtype, genome,
                   workers, chunk_size, column_chunk_size, progress)

    def import_d4(self, sources, *, workers=None, chunk_size=None,
                  column_chunk_size=None, progress=False) -> None:
        from ._native import import_d4

        import_d4(self.store_path, self.name, _norm_sources(sources),
                  workers, chunk_size, column_chunk_size, progress)

    def import_bigwig(self, sources, *, workers=None, chunk_size=None,
                      column_chunk_size=None, progress=False) -> None:
        from ._native import import_bigwig

        import_bigwig(self.store_path, self.name, _norm_sources(sources),
                      workers, chunk_size, column_chunk_size, progress)
