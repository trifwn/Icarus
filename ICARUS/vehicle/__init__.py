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

    isort:skip_file
"""

from .utils import DiscretizationType
from .utils import DistributionType
from .utils import SymmetryAxes

from .base_classes import (
    Mass,
    InertiaTensor,
    RigidBody,
)


from .control_surface import Aileron
from .control_surface import ControlSurface
from .control_surface import ControlType
from .control_surface import Elevator
from .control_surface import Flap
from .control_surface import NoControl
from .control_surface import Rudder
from .strip import Strip
from .surface import WingSurface
from .surface_connections import SurfaceConnection
from .wing_segment import WingSegment
from .wing import Wing
from .airplane import Airplane


__all__ = [
    "Airplane",
    "ControlSurface",
    "ControlType",
    "Aileron",
    "Flap",
    "Rudder",
    "Elevator",
    "NoControl",
    "Mass",
    "InertiaTensor",
    "RigidBody",
    "Wing",
    "Strip",
    "WingSegment",
    "WingSurface",
    "SurfaceConnection",
    "DiscretizationType",
    "DistributionType",
    "SymmetryAxes",
]
