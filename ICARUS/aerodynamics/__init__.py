"""=============================
Potential aerodynamics Module
=============================

.. toctree:
    :hidden:

    lifting_surfaces
    vorticity
    wing_lspt


.. module:: ICARUS.aerodynamics.potential
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for potential aerodynamic analysis on ICARUS objects.

.. currentmodule:: ICARUS.aerodynamics.potential

This package contains class and routines for potential aerodynamic analysis on ICARUS objects.
The package is divided in three libraries:

Lifting Surfaces Potential Theory
==================================
.. autosummary::
    :toctree:

    ICARUS.aerodynamics.potential.lifting_surfaces - Interface for solver class
    ICARUS.aerodynamics.potential.vorticity - Functions to solve the Biotsavart equation for different elements
    ICARUS.aerodynamics.potential.wing_lspt - A class modeling a wing for solving the lifting surfaces using panels and a potential theory formulation

"""

from . import biot_savart
from . import lifting_surfaces
from . import wing_lspt

__all__ = ["biot_savart", "lifting_surfaces", "wing_lspt"]
