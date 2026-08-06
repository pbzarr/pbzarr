"""pbzarr — Python wheel for PBZ (Per-Base Zarr) flat-layout stores."""

from . import _accessor as _accessor
from . import _datatree_accessor as _datatree_accessor
from ._native import PbzError
from ._open import open
from ._region import RegionQuery, parse_region
from ._write import (
    create_store,
    import_bed,
    import_bed_multi,
    import_bigwig,
    import_d4,
    stack,
)

__all__ = [
    "PbzError",
    "RegionQuery",
    "parse_region",
    "create_store",
    "import_d4",
    "import_bigwig",
    "import_bed",
    "import_bed_multi",
    "stack",
    "open",
]
