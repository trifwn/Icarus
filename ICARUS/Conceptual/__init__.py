"""
==========================================================
ICARUS package for Conceptual Design of Flying Vehicles
==========================================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.Conceptual.Criteria
    ICARUS.Conceptual.concept_airplane
    ICARUS.Conceptual.plane_concept
    ICARUS.Conceptual.wing_concept
    ICARUS.Conceptual.fuselage_concept

.. module:: ICARUS.Conceptual
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for conceptual design of flying vehicles.

.. currentmodule:: ICARUS.Conceptual

This package contains class and routines for conceptual design of flying vehicles. The conceptual analysis
is devided in two main parts:

Vehicle Modelling
--------------------------

.. autosummary::
    :toctree: generated/

    ICARUS.Conceptual.concept_airplane
    ICARUS.Conceptual.plane_concept
    ICARUS.Conceptual.wing_concept
    ICARUS.Conceptual.fuselage_concept

The Criteria part contains classes and routines for defining mission and performance requirements for the vehicle.

.. autosummary::
    :toctree: generated/

    ICARUS.Conceptual.Criteria

"""
from . import concept_airplane
from . import Criteria

__all__ = ['Criteria', 'concept_airplane']
