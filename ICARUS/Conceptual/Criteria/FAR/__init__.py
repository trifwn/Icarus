"""
===============================================================
ICARUS Module for adding FAR Criteria to the Conceptual Design
===============================================================

.. toctree:
    :hidden:

    ICARUS.Conceptual.Criteria.FAR.takeoff_criterion
    ICARUS.Conceptual.Criteria.FAR.climb_criterion
    ICARUS.Conceptual.Criteria.FAR.cruise_criterion
    ICARUS.Conceptual.Criteria.FAR.failed_approach_criterion
    ICARUS.Conceptual.Criteria.FAR.landing_criterion
    ICARUS.Conceptual.Criteria.FAR.range
    ICARUS.Conceptual.Criteria.FAR.useful_load
    ICARUS.Conceptual.Criteria.FAR.get_all_criteria
    ICARUS.Conceptual.Criteria.FAR.helper_functions

.. module:: ICARUS.Conceptual.Criteria.FAR
    :platform: Unix, Windows
    :synopsis: This package contains class and routines for defining the FAR criteria for the conceptual design of flying vehicles.

.. currentmodule:: ICARUS.Conceptual.Criteria.FAR

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
    'takeoff_criterion',
    'climb_criterion',
    'cruise_criterion',
    'failed_approach_criterion',
    'landing_criterion',
    'range',
    'useful_load',
    'get_all_criteria',
    'helper_functions',
]
