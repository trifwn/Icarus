"""
=============================
Potential Aerodynamics Module
=============================

.. toctree:
    :hidden:

    Potential.lifting_surfaces
    Potential.vorticity
    Potential.wing_lspt


.. module:: ICARUS.Aerodynamics.Potential
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for potential aerodynamic analysis on ICARUS objects.

.. currentmodule:: ICARUS.Aerodynamics.Potential

This package contains class and routines for potential aerodynamic analysis on ICARUS objects.
The package is divided in three libraries:

Lifting Surfaces Potential Theory
==================================
.. autosummary::
    :toctree:

    ICARUS.Aerodynamics.Potential.lifting_surfaces - Interface for solver class
    ICARUS.Aerodynamics.Potential.vorticity - Functions to solve the Biotsavart equation for different elements
    ICARUS.Aerodynamics.Potential.wing_lspt - A class modeling a wing for solving the lifting surfaces using panels and a potential theory formulation

"""
from . import lifting_surfaces
from . import vorticity
from . import wing_lspt

__all__ = ["lifting_surfaces", "vorticity", "wing_lspt"]
