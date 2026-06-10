"""pbzarr — Python wheel for PBZ (Per-Base Zarr) stores."""

from . import accessor  # registers .pbz on DataTree
from ._native import PbzError
from ._open import open
from ._pbzstore import PbzStore

__version__ = "0.2.0"

__all__ = [
    "PbzError",
    "PbzStore",
    "open",
    "__version__",
]
