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

    isort:skip_file
"""

from .figure_setup import create_single_subplot
from .figure_setup import create_subplots
from .figure_setup import flatten_axes
from .polar_plot import polar_plot
from .pre_existing_figure import pre_existing_figure
from .utils import colors_
from .utils import markers
from .utils import markers_str
from . import airfoil
from . import airplane
from . import avl
from . import gnvp

__all__ = [
    "airfoil",
    "airplane",
    "gnvp",
    "avl",
    "markers_str",
    "markers",
    "colors_",
    "flatten_axes",
    "create_subplots",
    "create_single_subplot",
    "pre_existing_figure",
    "polar_plot",
]
