"""
=======================================
ICARUS Vehicles Package
=======================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.Vehicle.strip
    ICARUS.Vehicle.wing_segment
    ICARUS.Vehicle.merged_wing
    ICARUS.Vehicle.plane
    ICARUS.Vehicle.surface_connections


.. module:: ICARUS.Vehicle
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for vehicle modelling and analysis.

.. currentmodule:: ICARUS.Vehicle

This package contains class and routines for vehicle modelling and analysis.

Strip
=====

.. autosummary::
    :toctree: generated/

    ICARUS.Vehicle.strip - Strip class definition

Wing Segment
============

.. autosummary::
    :toctree: generated/

    ICARUS.Vehicle.wing_segment - Wing Segment class definition

Merged Wing
===========

.. autosummary::
    :toctree: generated/

    ICARUS.Vehicle.merged_wing - Merged Wing class definition

Plane
========================

.. autosummary::
    :toctree: generated/

    ICARUS.Vehicle.plane - Plane class definition

Surface Connections
===================

.. autosummary::
    :toctree: generated/

    ICARUS.Vehicle.surface_connections - Surface Connections class definition

"""

__all__ = ['strip', 'wing_segment', "merged_wing", "plane", "surface_connections"]

from . import strip, wing_segment, merged_wing, plane, surface_connections
