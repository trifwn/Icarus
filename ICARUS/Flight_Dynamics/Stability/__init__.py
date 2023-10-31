"""
==================================================
ICARUS Flight Dynamics Stability Package
==================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Flight_Dynamics.Stability.longitudalFD
    ICARUS.Flight_Dynamics.Stability.lateralFD
    ICARUS.Flight_Dynamics.Stability.stability_derivatives

.. module:: ICARUS.Flight_Dynamics.Stability
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for flight dynamics stability computations.

.. currentmodule:: ICARUS.Flight_Dynamics.Stability

This package contains classes and routines for flight dynamics stability computations. The package is divided in the following files:

.. autosummary::
    :toctree:

    ICARUS.Flight_Dynamics.Stability.longitudalFD
    ICARUS.Flight_Dynamics.Stability.lateralFD
    ICARUS.Flight_Dynamics.Stability.stability_derivatives

"""
from . import lateralFD
from . import longitudalFD
from . import stability_derivatives

__all__ = ["lateralFD", "longitudalFD", "stability_derivatives"]
