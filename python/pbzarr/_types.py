"""Type aliases of the public API."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

from ._region_query import RegionQuery

PathLike: TypeAlias = str | os.PathLike[str]
Source: TypeAlias = PathLike | tuple[PathLike, str]
Codecs: TypeAlias = list[dict[str, Any]] | dict[str, Any]
RegionLike: TypeAlias = str | RegionQuery | tuple[str, int | None, int | None]
Intervals: TypeAlias = (
    RegionLike
    | pd.DataFrame
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | Sequence[str | RegionQuery]
    | Iterable[tuple[str, int, int]]
)
Reducer: TypeAlias = Literal[
    "sum", "mean", "min", "max", "count", "std", "var", "median", "quantile", "summit"
]
Fill: TypeAlias = float | bool | Literal["na"]
Chunks: TypeAlias = int | str | Mapping[str, int | str | tuple[int, ...] | None] | None
