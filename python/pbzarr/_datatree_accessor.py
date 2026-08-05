"""PBZ operations on native xarray DataTrees."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from ._regions import regions_dataset
from ._xarray import _compose_track_datasets, _normalize_track_names


@xr.register_datatree_accessor("pbz")
class PbzDataTreeAccessor:
    """PBZ operations for an opened collection."""

    def __init__(self, tree: xr.DataTree) -> None:
        self._tree = tree

    def dataset(self, tracks: Sequence[str] | None = None) -> xr.Dataset:
        """Compose selected same-genome tracks without aligning their labels."""
        names = (
            list(self._tree.children)
            if tracks is None
            else _normalize_track_names(tracks, self._tree.children)
        )
        if not names:
            dataset = self._tree.to_dataset(inherit=False)
        else:
            datasets = {
                name: self._tree[name].to_dataset(inherit=False) for name in names
            }
            dataset = _compose_track_datasets(datasets)
        dataset.set_close(self._tree.close)
        return dataset

    def regions(
        self,
        intervals,
        *,
        tracks: Sequence[str] | None = None,
    ) -> xr.Dataset:
        """Pack intervals from selected same-genome tracks along position.

        Dask-backed values remain lazy. With ``chunks=None``, selected pieces
        are gathered eagerly and the complete packed result must fit in RAM.
        """
        return regions_dataset(self.dataset(tracks), intervals)
