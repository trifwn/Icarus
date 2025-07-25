"""
Xfoil solver module for ICARUS.

isort: skip_file
"""

from . import analyses
from .xfoil import XfoilSolverParameters
from .xfoil import Xfoil
from .xfoil import XfoilAseq
from .xfoil import XfoilAseqInput
from .xfoil import XfoilAseqResetBL

__all__ = [
    "XfoilSolverParameters",
    "Xfoil",
    "XfoilAseq",
    "XfoilAseqResetBL",
    "XfoilAseqInput",
    "analyses",
]
