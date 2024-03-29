from typing import Any

import numpy as np

from .helper_functions import get_climb_rate
from .helper_functions import shape_like
from ICARUS.core.types import FloatArray

# FAR 4 -> 25.114


def far_4_climb(
    no_of_engines: int,
    cl_2: float,
    cd: float,
    AR: float,
    e: float,
    wing_loading: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Returns the thrust loading for a given number of engines, cl_2, cd, AR and e

    Args:
        no_of_engines (int): Number of engines
        cl_2 (float): Lift coefficient at Climb Segment 35 to 400 ft
        cd (float): Drag coefficient
        AR (float): Aspect ratio
        e (float): Oswald efficiency factor

    Returns:
        tuple: Wing loading, Thrust loading
    """
    lift_o_drag: float = cl_2 / (cd + cl_2**2 / (np.pi * AR * e))
    N: int = no_of_engines
    g1: float = get_climb_rate(N)

    thrust_loading: float = N / (N - 1) / (g1 + lift_o_drag)
    thrust_loading_arr: FloatArray = shape_like(thrust_loading, wing_loading)
    return wing_loading, thrust_loading_arr
