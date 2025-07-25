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
    airfoil_geometry - JAX-compatible Airfoil class with JIT compilation and automatic differentiation

Airfoil Polars Analysis
=======================

.. autosummary::
    :toctree: generated/

    airfoil_polars - Airfoil Polars class definition



"""

from .airfoil import Airfoil
from .core.airfoil_geometry import AirfoilGeometry
from .metrics import AirfoilData
from .metrics import AirfoilOperatingConditions
from .metrics import AirfoilOperatingPointMetrics
from .metrics import AirfoilPolar
from .metrics import AirfoilPolarMap
from .metrics import AirfoilPressure
from .metrics import PolarNotAccurate
from .metrics import ReynoldsNotIncluded
from .naca4 import NACA4
from .naca5 import NACA5

__all__ = [
    # Airfoil Modelling
    "Airfoil",
    "AirfoilGeometry",
    "NACA4",
    "NACA5",
    # Aerodynamic Data Classes
    "AirfoilOperatingPointMetrics",
    "AirfoilPressure",
    "AirfoilOperatingConditions",
    # Result Classes
    "AirfoilData",
    "AirfoilPolar",
    "AirfoilPolarMap",
    # Exceptions
    "PolarNotAccurate",
    "ReynoldsNotIncluded",
]
