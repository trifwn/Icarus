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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle


def get_p_RdBl_cmap() -> colors.LinearSegmentedColormap:
    """p_RdBl red-blue colormap."""
    cdict: dict[Literal['red', 'green', 'blue', 'alpha'], Sequence[tuple[float, float, float]]] = {
        "red": [(0.0, 217, 217), (0.5, 242, 242), (1.0, 65, 65)],
        "green": [(0.0, 58, 58), (0.5, 242, 242), (1.0, 124, 124)],
        "blue": [(0.0, 70, 70), (0.5, 242, 242), (1.0, 167, 167)],
    }
    # Normalize
    n_cdict: dict[Literal['red', 'green', 'blue', 'alpha'], Sequence[tuple[float, float, float]]] = {
        color: [(x[0], x[1] / 255.0, x[2] / 255.0) for x in scale] for color, scale in cdict.items()
    }
    return colors.LinearSegmentedColormap("p_RdBl", n_cdict)


from matplotlib import colormaps

viridis = plt.get_cmap('viridis')
coolwarm = colormaps.get_cmap("coolwarm")

viridis_listed = ListedColormap(viridis(np.arange(256)))
colors_ = viridis_listed

markers_str: list[str | int] = list(MarkerStyle.markers.keys())
markers = [MarkerStyle(marker) for marker in markers_str]

from . import airfoil
from . import airplane

__all__ = ["airfoil", "airplane"]

from .figure_setup import create_single_subplot
from .figure_setup import create_subplots
from .figure_setup import flatten_axes
