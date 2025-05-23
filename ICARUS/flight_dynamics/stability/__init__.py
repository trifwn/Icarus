"""==================================================
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

from .lateral import lateral_stability_finite_differences
from .longitudal import longitudal_stability_finite_differences
from .state_space import LateralStateSpace, LongitudalStateSpace, StateSpace
from .stability_derivatives import StabilityDerivativesDS

__all__ = [
    "lateral_stability_finite_differences",
    "longitudal_stability_finite_differences",
    "LateralStateSpace",
    "LongitudalStateSpace",
    "StateSpace",
    "StabilityDerivativesDS",
]
