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
    ICARUS.aero.- A class modeling a wing for solving the lifting surfaces using panels and a potential theory formulation

    isort:skip_file
"""

from .utils import panel_cp, panel_cp_normal
from .strip_loads import StripLoads
from .lspt_plane import LSPT_Plane
from .aerodynamic_state import AerodynamicState
from .aerodynamic_loads import AerodynamicLoads
from .aerodynamic_results import AerodynamicResults

from . import post_process
from . import vlm
from . import vpm

__all__ = [
    "panel_cp",
    "panel_cp_normal",
    "AerodynamicState",
    "AerodynamicLoads",
    "AerodynamicResults",
    "LSPT_Plane",
    "StripLoads",
    "vlm",
    "vpm",
    "post_process",
]
