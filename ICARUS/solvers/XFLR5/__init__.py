from .dynamic_analysis import xflr_eigs
from .parser import parse_xfl_project
from .read_xflr5_polars import parse_airfoil_name
from .read_xflr5_polars import read_XFLR5_airfoil_polars
from .read_xflr5_polars import read_XFLR5_airplane_polars

__all__ = [
    "parse_xfl_project",
    "read_XFLR5_airfoil_polars",
    "read_XFLR5_airplane_polars",
    "parse_airfoil_name",
    "xflr_eigs",
]
