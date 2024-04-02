"""
===============================================================
ICARUS Module for adding FAR Criteria to the Conceptual Design
===============================================================

.. toctree:
    :hidden:

    ICARUS.conceptual.criteria.FAR.takeoff_criterion
    ICARUS.conceptual.criteria.FAR.climb_criterion
    ICARUS.conceptual.criteria.FAR.cruise_criterion
    ICARUS.conceptual.criteria.FAR.failed_approach_criterion
    ICARUS.conceptual.criteria.FAR.landing_criterion
    ICARUS.conceptual.criteria.FAR.range
    ICARUS.conceptual.criteria.FAR.useful_load
    ICARUS.conceptual.criteria.FAR.get_all_criteria
    ICARUS.conceptual.criteria.FAR.helper_functions

.. module:: ICARUS.conceptual.criteria.FAR
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining the FAR criteria for the conceptual design of flying vehicles.

.. currentmodule:: ICARUS.conceptual.criteria.FAR

FAR CRITERIA
--------------

This package contains class and routines for defining the FAR criteria for the conceptual design of flying vehicles.
The FAR criteria are defined in the following files:

.. autosummary::
    :toctree:

    takeoff_criterion
    climb_criterion
    cruise_criterion
    failed_approach_criterion
    landing_criterion
    range

Helper Functions
----------------------

.. autosummary::
    :toctree:

    useful_load
    get_all_criteria
    helper_functions

"""
from . import climb_criterion
from . import cruise_criterion
from . import failed_approach_criterion
from . import get_all_criteria
from . import helper_functions
from . import landing_criterion
from . import range
from . import takeoff_criterion
from . import useful_load

__all__ = [
    "takeoff_criterion",
    "climb_criterion",
    "cruise_criterion",
    "failed_approach_criterion",
    "landing_criterion",
    "range",
    "useful_load",
    "get_all_criteria",
    "helper_functions",
]
