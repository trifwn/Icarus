"""
=================================================
ICARUS Airfoil Visualization and Graphics Module
=================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Visualization.db_polars
    ICARUS.Visualization.db_reynolds
    ICARUS.Visualization.f2w_pressure

.. module:: ICARUS.Visualization.airfoil
    :platform: Unix, Windows
    :synopsis: This module contains classes and routines for visualization of airfoil data.

.. currentmodule:: ICARUS.Visualization.airfoil

This module contains classes and routines for visualization of airfoil data.

Database Visualization
----------------------

To inspect the data stored in the Database data there are 2 main routines:

.. autosummary::
    :toctree: generated/

    ICARUS.Visualization.airfoil.db_polars - Plots the polars from the database
    ICARUS.Visualization.airfoil.db_reynolds - Plots the polars from the database for a specific Reynolds number

F2W Visualization
-------------------

For the visualization of the results of the F2W solver, the following routines are available:

.. autosummary::
    :toctree: generated/

    ICARUS.Visualization.airfoil.f2w_pressure - Plots the pressure distribution on the airfoil

"""
from typing import Any
from typing import Callable

from . import db_polars
from . import db_reynolds
from . import f2w_pressure

__all__ = ['db_polars', 'db_reynolds', 'f2w_pressure']

# Get all the visualization functions
from .db_polars import plot_airfoil_polars
from .db_reynolds import plot_airfoil_reynolds

# from .f2w_pressure import plot_angle_cp, plot_multiple_cps < Deprecated

__functions__: list[Callable[..., Any]] = [plot_airfoil_polars, plot_airfoil_reynolds]
