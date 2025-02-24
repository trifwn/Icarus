"""=================================================
ICARUS Airfoil Visualization and Graphics Module
=================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.visualization.airfoil_polars
    ICARUS.visualization.airfoil_reynolds
    ICARUS.visualization.f2w_pressure

.. module:: ICARUS.visualization.airfoil
    :platform: Unix, Windows
    :synopsis: This module contains classes and routines for visualization of airfoil data.

.. currentmodule:: ICARUS.visualization.airfoil

This module contains classes and routines for visualization of airfoil data.

Database Visualization
----------------------

To inspect the data stored in the Database data there are 2 main routines:

.. autosummary::
    :toctree: generated/

    ICARUS.visualization.airfoil.airfoil_polars - Plots the polars from the database
    ICARUS.visualization.airfoil.airfoil_reynolds - Plots the polars from the database for a specific Reynolds number

F2W Visualization
-------------------

For the visualization of the results of the F2W solver, the following routines are available:

.. autosummary::
    :toctree: generated/

    ICARUS.visualization.airfoil.f2w_pressure - Plots the pressure distribution on the airfoil

"""

from typing import Any
from typing import Callable

from . import airfoil_polars
from . import airfoil_reynolds
from . import f2w_pressure

__all__ = ["airfoil_polars", "airfoil_reynolds", "f2w_pressure"]

# Get all the visualization functions
from .airfoil_polars import plot_airfoil_polars
from .airfoil_reynolds import plot_airfoils_at_reynolds

# from .f2w_pressure import plot_angle_cp, plot_multiple_cps < Deprecated

__functions__: list[Callable[..., Any]] = [plot_airfoil_polars, plot_airfoils_at_reynolds]
