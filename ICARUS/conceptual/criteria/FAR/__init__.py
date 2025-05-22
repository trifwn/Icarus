"""===============================================================
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
    ICARUS.conceptual.criteria.FAR

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

from .climb_criterion import far_4_climb
from .cruise_criterion import far_5_cruise_speed
from .failed_approach_criterion import far_2_failed_approach
from .get_all_criteria import get_all_far_criteria
from .helper_functions import drag_coeff_skin
from .helper_functions import get_climb_rate
from .helper_functions import get_climb_rate_failed_approach
from .helper_functions import lift_drag_min_drag
from .landing_criterion import far_1_landing
from .landing_criterion import far_inverse_landing_criterion_cl_max
from .landing_criterion import far_inverse_landing_criterion_cl_max2
from .range import range_criterion
from .takeoff_criterion import far_3_takeoff
from .useful_load import usefull_load_criterion

__all__ = [
    # FAR criteria
    "get_all_far_criteria",
    "far_1_landing",
    "far_2_failed_approach",
    "far_3_takeoff",
    "far_4_climb",
    "far_5_cruise_speed",
    "range_criterion",
    "usefull_load_criterion",
    # FAR inverse criteria
    "far_inverse_landing_criterion_cl_max",
    "far_inverse_landing_criterion_cl_max2",
    # FAR helper functions
    "drag_coeff_skin",
    "get_climb_rate",
    "get_climb_rate_failed_approach",
    "lift_drag_min_drag",
]
