"""=============================
Potential aerodynamics Module
=============================

.. toctree:
    :hidden:

    lifting_surfaces
    vorticity
    wing_lspt


.. module:: ICARUS.aero.potential
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for potential aerodynamic analysis on ICARUS objects.

.. currentmodule:: ICARUS.aero.potential

This package contains class and routines for potential aerodynamic analysis on ICARUS objects.
The package is divided in three libraries:

Lifting Surfaces Potential Theory
==================================
.. autosummary::
    :toctree:

    ICARUS.aero.potential.lifting_surfaces - Interface for solver class
    ICARUS.aero.potential.vorticity - Functions to solve the Biotsavart equation for different elements
    ICARUS.aero.plane_lspt - A class modeling a wing for solving the lifting surfaces using panels and a potential theory formulation

"""

from . import post_process
from . import vlm
from . import vpm
from .plane_lspt import LSPT_Plane
from .strip_data import StripData

__all__ = [
    "LSPT_Plane",
    "StripData",
    "vlm",
    "vpm",
    "post_process",
]
