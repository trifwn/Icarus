"""
=================================
ICARUS Mission Modelling Package
=================================

.. toctree: generated/
    :hidden:

    ICARUS.mission.segment
    ICARUS.mission.trajectory
    ICARUS.mission.mission_analysis
    ICARUS.mission.mission_definition
    ICARUS.mission.mission_performance
    ICARUS.mission.mission_vehicle

.. module:: ICARUS.mission
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for mission modelling.

.. currentmodule:: ICARUS.mission

This package contains classes and routines for mission modelling. The package defines 3 basic components:

.. autosummary::
    :toctree: generated/

    mission_analysis
    mission_definition
    mission_vehicle
    mission_performance

Additionally, the package contains the following modules:

.. autosummary::
    :toctree: generated/

    segment
    trajectory



"""
from . import mission
from . import mission_analysis
from . import mission_performance
from . import mission_vehicle
from . import segment
from . import trajectory

__all__ = [
    "mission_analysis",
    "mission",
    "segment",
    "trajectory",
    "mission_performance",
    "mission_vehicle",
]
