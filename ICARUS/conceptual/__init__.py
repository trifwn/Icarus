"""==========================================================
ICARUS package for Conceptual Design of Flying vehicles
==========================================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.conceptual.criteria
    ICARUS.conceptual.concept_airplane
    ICARUS.conceptual.plane_concept

.. module:: ICARUS.conceptual
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for conceptual design of flying vehicles.

.. currentmodule:: ICARUS.conceptual

This package contains class and routines for conceptual design of flying vehicles. The conceptual analysis
is devided in two main parts:

Vehicle Modelling
--------------------------

.. autosummary::
    :toctree: generated/

    ICARUS.conceptual.concept_airplane

The Criteria part contains classes and routines for defining mission and performance requirements for the vehicle.

.. autosummary::
    :toctree: generated/

    ICARUS.conceptual.criteria

"""

from . import criteria
from .concept_airplane import ConceptAirplane
from .concept_engine import ConceptEngine

__all__ = [
    "ConceptAirplane",
    "ConceptEngine",
    "criteria",
]
