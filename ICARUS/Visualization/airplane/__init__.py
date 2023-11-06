"""
=================================================
ICARUS Airplane Visualization and Graphics Module
=================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Visualization.gnvp_convergence
    ICARUS.Visualization.db_polars
    ICARUS.Visualization.gnvp_sensitivity
    ICARUS.Visualization.gnvp_strips
    ICARUS.Visualization.gnvp_wake

.. module:: ICARUS.Visualization.airplane
    :platform: Unix, Windows
    :synopsis: This module contains classes and routines for visualization of airplane data.

.. currentmodule:: ICARUS.Visualization.airfoil

This module contains classes and routines for visualization of airplane data.

Database Visualization
----------------------

To see the results stored in the database, the following routines are available:

.. autosummary::
    :toctree: generated/

    ICARUS.Visualization.airplane.db_polars - Plots the polars from the database

GNVP Visualization
-------------------

There are also routines for the visualization of the results of the GNVP results:

.. autosummary::
    :toctree: generated/

    ICARUS.Visualization.airplane.gnvp_convergence - Plots the convergence of the GNVP algorithm
    ICARUS.Visualization.airplane.gnvp_sensitivity - Plots the sensitivity of the GNVP algorithm
    ICARUS.Visualization.airplane.gnvp_strips - Plots the strip analysis of the GNVP algorithm
    ICARUS.Visualization.airplane.gnvp_wake - Plots the wake analysis of the GNVP algorithm


"""
from . import db_polars
from . import gnvp_convergence
from . import gnvp_sensitivity
from . import gnvp_strips
from . import gnvp_wake

__all__ = ["db_polars", "gnvp_convergence", "gnvp_sensitivity", "gnvp_strips", "plot_gnvp_wake"]

# Get all the visualization functions
from typing import Any, Callable
from .db_polars import plot_airplane_polars
from .gnvp_convergence import plot_convergence
from .gnvp_sensitivity import plot_sensitivity
from .gnvp_strips import gnvp_strips_2d, gnvp_strips_3d
from .gnvp_wake import plot_gnvp_wake

__functions__: list[Callable[..., Any]] = [
    plot_airplane_polars,
    plot_convergence,
    plot_sensitivity,
    gnvp_strips_2d,
    gnvp_strips_3d,
    plot_gnvp_wake,
]
