"""============================================
ICARUS Visualization Module
============================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.visualization.airfoil
    ICARUS.visualization.airplane

.. module:: ICARUS.visualization
    :platform: Unix, Windows
    :synopsis: This module contains classes and routines for visualization.

.. currentmodule:: ICARUS.visualization

This module contains classes and routines for visualization. The module is divided in two modules:

.. autosummary::
    :toctree: generated/

    ICARUS.visualization.airfoil - Airfoil visualization
    ICARUS.visualization.airplane - Airplane visualization

"""
from .utils import colors_, markers_str, markers
from . import airfoil
from . import airplane
from . import gnvp
from . import f2w
from . import avl

__all__ = [
    "airfoil",
    "airplane",
    "gnvp",
    "f2w",
    "avl",
    "markers_str",
    "markers",
    "colors_",
]
