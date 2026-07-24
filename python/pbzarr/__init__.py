"""pbzarr — Python wheel for PBZ (Per-Base Zarr) flat-layout stores."""

from . import _accessor  # noqa: F401 - registers the .pbz Dataset accessor
from ._native import PbzError
from ._open import open
from ._pbzstore import PbzStore
from ._read import RegionBlocks
from ._stack import stack
from ._store import create_store
from ._track import Track

__version__ = "0.4.0"

__all__ = [
    "PbzError",
    "PbzStore",
    "Track",
    "RegionBlocks",
    "create_store",
    "stack",
    "open",
    "__version__",
]
