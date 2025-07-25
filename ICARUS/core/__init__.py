"""===========================================
ICARUS Core package
===========================================

.. toctree: generated/
    :hidden:

    ICARUS.core.formatting
    ICARUS.core.rotate
    ICARUS.core.struct
    ICARUS.core.file_tail
    ICARUS.core.types
    ICARUS.core.units

.. module:: ICARUS.core
    :platform: Unix, Windows
    :synopsis: This package contains core utility routines for ICARUS.

.. currentmodule:: ICARUS.core

The ICARUS Core package contains routines and classes that are used throughtout ICARUS but are not specific to any particular module. The package is divided in the following files:

.. autosummary::
    :toctree: generated/

    ICARUS.core.formatting
    ICARUS.core.rotate
    ICARUS.core.struct
    ICARUS.core.file_tail
    ICARUS.core.types
    ICARUS.core.units
"""

from . import math
from . import serialization
from . import types
from . import units
from . import utils
from .base_types import Optimizable
from .base_types import Struct

__all__ = [
    "Struct",
    "Optimizable",
    "utils",
    "math",
    "serialization",
    "types",
    "units",
]
