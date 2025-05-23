"""============================================
ICARUS Flight Dynamics Package
============================================

.. toctree: generated/
    :hidden:

    ICARUS.flight_dynamics.disturbances
    ICARUS.flight_dynamics.perturbations
    ICARUS.flight_dynamics.state
    ICARUS.flight_dynamics.trim
    ICARUS.flight_dynamics.stability

.. module:: ICARUS.flight_dynamics
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for flight dynamics modelling.

.. currentmodule:: ICARUS.flight_dynamics

Flight Dynamics Modelling
----------------------------

This package contains classes and routines for flight dynamics modelling.
To model the flight dynamics of an aircraft, the following classes are defined:

.. autosummary::
    :toctree: generated/

    ICARUS.flight_dynamics.disturbances
    ICARUS.flight_dynamics.perturbations
    ICARUS.flight_dynamics.state
    ICARUS.flight_dynamics.trim


Stability Analysis
-----------------------------

Around a given trim point, the stability derivatives are computed and the stability of the aircraft is analysed using Finite
Difference methods in the following package:

.. autosummary::
    :toctree: generated/

    ICARUS.flight_dynamics.stability
"""

from .stability import (
    lateral_stability_finite_differences,
    longitudal_stability_finite_differences,
    LateralStateSpace,
    LongitudalStateSpace,
    StateSpace,
    StabilityDerivativesDS,
)

from .perturbations import longitudal_pertrubations, lateral_pertrubations
from .disturbances import Disturbance
from .trim import trim_state, Trim, TrimNotPossible, TrimOutsidePolars
from .state import State

__all__ = [
    "lateral_stability_finite_differences",
    "longitudal_stability_finite_differences",
    "LateralStateSpace",
    "LongitudalStateSpace",
    "StateSpace",
    "StabilityDerivativesDS",
    "longitudal_pertrubations",
    "lateral_pertrubations",
    "Disturbance",
    "trim_state",
    "Trim",
    "TrimNotPossible",
    "TrimOutsidePolars",
    "State",
]
