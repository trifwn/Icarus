"""
==================================================
ICARUS Flight Dynamics Stability Package
==================================================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.flight_dynamics.stability.longitudal
    ICARUS.flight_dynamics.stability.lateral
    ICARUS.flight_dynamics.stability.stability_derivatives

.. module:: ICARUS.flight_dynamics.stability
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for flight dynamics stability computations.

.. currentmodule:: ICARUS.flight_dynamics.stability

This package contains classes and routines for flight dynamics stability computations. The package is divided in the following files:

.. autosummary::
    :toctree:

    ICARUS.flight_dynamics.stability.longitudal
    ICARUS.flight_dynamics.stability.lateral
    ICARUS.flight_dynamics.stability.stability_derivatives

"""

from . import lateral
from . import longitudal
from . import stability_derivatives

__all__ = ["lateral", "longitudal", "stability_derivatives"]
