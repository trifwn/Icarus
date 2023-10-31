"""
=================================
ICARUS Mission Modelling Package
=================================

.. toctree: generated/
    :hidden:

    ICARUS.Mission.mission_analysis
    ICARUS.Mission.mission_definition
    ICARUS.Mission.mission_segment
    ICARUS.Mission.mission_path
    ICARUS.Mission.mission_performance

.. module:: ICARUS.Mission
    :platform: Unix, Windows
    :synopsis: This package contains classes and routines for mission modelling.

.. currentmodule:: ICARUS.Mission

This package contains classes and routines for mission modelling. The package defines five components:

.. autosummary::
    :toctree: generated/

    ICARUS.Mission.mission_analysis
    ICARUS.Mission.mission_definition
    ICARUS.Mission.mission_segment
    ICARUS.Mission.mission_path
    ICARUS.Mission.mission_performance

"""
from . import mission_analysis
from . import mission_definition
from . import mission_path
from . import mission_performance
from . import mission_segment

__all__ = ["mission_analysis", "mission_definition", "mission_segment", "mission_path", "mission_performance"]
