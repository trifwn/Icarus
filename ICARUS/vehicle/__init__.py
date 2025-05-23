"""=======================================
ICARUS vehicles Package
=======================================

.. toctree: generated/
    :hidden:
    :noindex:

    ICARUS.vehicle.strip
    ICARUS.vehicle.surface
    ICARUS.vehicle.merged_wing
    ICARUS.vehicle.airplane
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

    ICARUS.vehicle.airplane - Plane class definition

Surface Connections
===================

.. autosummary::
    :toctree: generated/

    ICARUS.vehicle.surface_connections - Surface Connections class definition

"""

from .airplane import Airplane
from .control_surface import ControlSurface, ControlType, Aileron, Flap, Rudder, Elevator, NoControl
from .point_mass import PointMass
from .wing import Wing
from .strip import Strip
from .wing_segment import WingSegment
from .surface import WingSurface
from .surface_connections import SurfaceConnection
from .utils import DiscretizationType, DistributionType, SymmetryAxes


__all__ = [
    "Airplane",
    "ControlSurface",
    "ControlType",
    "Aileron",
    "Flap",
    "Rudder",
    "Elevator",
    "NoControl",
    "PointMass",
    "Wing",
    "Strip",
    "WingSegment",
    "WingSurface",
    "SurfaceConnection",
    "DiscretizationType",
    "DistributionType",
    "SymmetryAxes",
]
