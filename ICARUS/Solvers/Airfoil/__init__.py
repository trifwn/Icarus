"""
===============================
ICARUS Airfoil Solver Package
===============================

.. toctree:
    :hidden:
    :noindex:

    ICARUS.Solvers.Airfoil.f2w_section
    ICARUS.Solvers.Airfoil.open_foam
    ICARUS.Solvers.Airfoil.xfoil

.. module:: ICARUS.Solvers.Airfoil
    :platform: Unix, Windows
    :synopsis: This package contains object definitions for different solvers that can be used in ICARUS.

.. currentmodule:: ICARUS.Solvers.Airfoil

This package contains object definitions for different solvers that can be used in ICARUS to compute the
aerodynamics of Airfoils. The currently configured solvers are:

.. autosummary::
    :toctree:

    f2w_section - F2W Section solver
    open_foam - OpenFOAM solver
    xfoil - XFOIL solver

"""
from . import f2w_section
from . import open_foam
from . import xfoil

__all__ = ["f2w_section", "open_foam", "xfoil"]
