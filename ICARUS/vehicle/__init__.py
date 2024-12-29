"""=======================================
ICARUS vehicles Package
=======================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.vehicle.strip
    ICARUS.vehicle.surface
    ICARUS.vehicle.merged_wing
    ICARUS.vehicle.plane
    ICARUS.vehicle.surface_connections


.. module:: ICARUS.vehicle
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for vehicle modelling and analysis.

.. currentmodule:: ICARUS.vehicle

This package contains class and routines for vehicle modelling and analysis.

Strip
=====

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.strip - Strip class definition

Wing Segment
============

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.surface - Wing Segment class definition

Merged Wing
===========

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.merged_wing - Merged Wing class definition

Plane
========================

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.plane - Plane class definition

Surface Connections
===================

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.surface_connections - Surface Connections class definition

"""

__all__ = ["merged_wing", "plane", "strip", "surface", "surface_connections"]

from . import merged_wing
from . import plane
from . import strip
from . import surface
from . import surface_connections
