"""
=================================
ICARUS Mission Modelling Package
=================================

.. toctree: generated/
    :hidden:

    ICARUS.Mission.Segment
    ICARUS.Mission.Trajectory
    ICARUS.Mission.mission_analysis
    ICARUS.Mission.mission_definition
    ICARUS.Mission.mission_performance
    ICARUS.Mission.mission_vehicle

.. module:: ICARUS.Mission
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for mission modelling.

.. currentmodule:: ICARUS.Mission

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

    Segment
    Trajectory



"""
from . import mission_analysis
from . import mission_definition
from . import mission_performance
from . import mission_vehicle
from . import Segment
from . import Trajectory

__all__ = ["mission_analysis", "mission_definition", "Segment", "Trajectory", "mission_performance", "mission_vehicle"]
