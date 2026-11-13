"""pbzarr: per-base genomic signal in Zarr v3 stores."""

from importlib.metadata import version as _version

from ._dataset import create, open, write
from ._genome import interval_mask, reduce_regions, region
from ._native import PbzError
from ._reference import Reference
from ._region_query import RegionQuery, parse_region
from ._write import import_bam, import_bed, import_bigwig, import_d4

__version__: str = _version("pbzarr")

__all__ = [
    "PbzError",
    "Reference",
    "RegionQuery",
    "create",
    "import_bam",
    "import_bed",
    "import_bigwig",
    "import_d4",
    "interval_mask",
    "open",
    "parse_region",
    "reduce_regions",
    "region",
    "write",
]
