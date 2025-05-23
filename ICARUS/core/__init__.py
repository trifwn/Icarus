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

from . import base_types
from . import math
from . import serialization
from . import types
from . import units
from . import utils

__all__ = [
    "base_types",
    "utils",
    "math",
    "serialization",
    "types",
    "units",
]
