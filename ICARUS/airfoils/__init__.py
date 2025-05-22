"""===================================================
ICARUS airfoils modelling and analysis package
===================================================

.. toctree: generated/
    :hidden:

    airfoils.airfoil
    airfoils.airfoil_polars

.. module:: ICARUS.airfoils
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for airfoil modelling and analysis.

.. currentmodule:: ICARUS.airfoils

This package contains class and routines for airfoil modelling and analysis. The package is divided in two modules:


Airfoil Modelling
==================

.. autosummary::
    :toctree: generated/

    airfoil - Airfoil class definition

Airfoil Polars Analysis
=======================

.. autosummary::
    :toctree: generated/

    airfoil_polars - Airfoil Polars class definition



"""

from .airfoil import Airfoil
from .airfoil_polars import AirfoilData
from .airfoil_polars import AirfoilPolars
from .airfoil_polars import PolarNotAccurate
from .airfoil_polars import ReynoldsNotIncluded
from .naca4 import NACA4
from .naca5 import NACA5

__all__ = [
    # Airfoil Modelling
    "Airfoil",
    "NACA4",
    "NACA5",
    # Result Classes
    "AirfoilData",
    "AirfoilPolars",
    # Exceptions
    "PolarNotAccurate",
    "ReynoldsNotIncluded",
]
