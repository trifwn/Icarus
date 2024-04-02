"""
============================================
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
from typing import Literal
from typing import Sequence

import matplotlib
from matplotlib import colors
from matplotlib.markers import MarkerStyle

cdict: dict[Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, float, float]]] = {
    "red": ((0.0, 0.22, 0.0), (0.5, 1.0, 1.0), (1.0, 0.89, 1.0)),
    "green": ((0.0, 0.49, 0.0), (0.5, 1.0, 1.0), (1.0, 0.12, 1.0)),
    "blue": ((0.0, 0.72, 0.0), (0.5, 0.0, 0.0), (1.0, 0.11, 1.0)),
}

colors_ = colors.LinearSegmentedColormap("custom", cdict)

markers_str: list[str | int] = list(MarkerStyle.markers.keys())
markers = [MarkerStyle(marker) for marker in markers_str]

from . import airfoil
from . import airplane

__all__ = ["airfoil", "airplane"]
