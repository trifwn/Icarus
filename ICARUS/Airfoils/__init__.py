"""
===================================================
ICARUS Airfoils modelling and analysis package
===================================================

.. toctree: generated/
    :hidden:

    Airfoils.airfoil
    Airfoils.airfoil_polars

.. module:: ICARUS.Airfoils
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for airfoil modelling and analysis.

.. currentmodule:: ICARUS.Airfoils

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

__all__ = ['airfoil', 'airfoil_polars']
