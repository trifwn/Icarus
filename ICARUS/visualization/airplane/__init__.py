"""=================================================
ICARUS Airplane Visualization and Graphics Module
=================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.visualization.airplane.gnvp_convergence
    ICARUS.visualization.airplane.airplane_polars
    ICARUS.visualization.airplane.gnvp_sensitivity
    ICARUS.visualization.airplane.gnvp_strips
    ICARUS.visualization.airplane.gnvp_wake

.. module:: ICARUS.visualization.airplane
    :platform: Unix, Windows
    :synopsis: This module contains classes and routines for visualization of airplane data.

.. currentmodule:: ICARUS.visualization.airplane

This module contains classes and routines for visualization of airplane data.

Database Visualization
----------------------

To see the results stored in the database, the following routines are available:

.. autosummary::
    :toctree: generated/

    ICARUS.visualization.airplane.airplane_polars - Plots the polars from the database

GNVP Visualization
-------------------

There are also routines for the visualization of the results of the GNVP results:

.. autosummary::
    :toctree: generated/

    ICARUS.visualization.airplane.gnvp_convergence - Plots the convergence of the GNVP algorithm
    ICARUS.visualization.airplane.gnvp_sensitivity - Plots the sensitivity of the GNVP algorithm
    ICARUS.visualization.airplane.gnvp_strips - Plots the strip analysis of the GNVP algorithm
    ICARUS.visualization.airplane.gnvp_wake - Plots the wake analysis of the GNVP algorithm


"""

# Get all the visualization functions
from typing import Any
from typing import Callable

from .airplane_polars import plot_airplane_polars
from .cg_investigation import cg_investigation

__all__ = [
    "plot_airplane_polars",
    "cg_investigation",
]

__functions__: list[Callable[..., Any]] = [plot_airplane_polars, cg_investigation]
