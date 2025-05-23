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

    isort:skip_file
"""

from .disturbances import Disturbance
from .perturbations import lateral_pertrubations
from .perturbations import longitudal_pertrubations
from .stability import LateralStateSpace
from .stability import LongitudalStateSpace
from .stability import StabilityDerivativesDS
from .stability import StateSpace
from .stability import lateral_stability_finite_differences
from .stability import longitudal_stability_finite_differences
from .state import State
from .trim import Trim
from .trim import TrimNotPossible
from .trim import TrimOutsidePolars
from .trim import trim_state

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
