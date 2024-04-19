"""
===================================================
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

from . import airfoil
from . import airfoil_polars

__all__ = ["airfoil", "airfoil_polars"]
