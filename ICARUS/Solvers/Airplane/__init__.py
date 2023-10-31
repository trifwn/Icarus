"""
===============================
ICARUS Airplane Solver Package
===============================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Solvers.Airplane.gnvp3
    ICARUS.Solvers.Airplane.gnvp7
    ICARUS.Solvers.Airplane.lspt

.. module:: ICARUS.Solvers.Airplane
    :platform: Unix, Windows
    :synopsis: This package contains object definitions for different solvers that can be used in ICARUS to compute airplane aerodynamics.

.. currentmodule:: ICARUS.Solvers.Airplane

This package contains object definitions for different solvers that can be used in ICARUS to compute the
aerodynamics of Airplanes. The currently configured solvers are:

.. autosummary::
    :toctree:

    gnvp3 - GNVP3 solver
    gnvp7 - GNVP7 solver
    lspt - LSPT solver
"""
from . import gnvp3
from . import gnvp7
from . import lspt

__all__ = ["gnvp3", "gnvp7", "lspt"]
