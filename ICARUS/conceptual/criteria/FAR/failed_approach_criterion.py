import numpy as np

from ICARUS.core.types import FloatArray

from .helper_functions import get_climb_rate_failed_approach
from .helper_functions import shape_like


# FAR 2 -> 25.112
def far_2_failed_approach(
    no_of_engines: int,
    cl_app: float,
    cd: float,
    AR: float,
    e: float,
    wing_loading: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Returns the thrust loading for a given number of engines, cl_max, cd, AR and e

    Args:
        no_of_engines (int): Number of engines
        cl_max (float): Cl_max for landing
        cd (float): Cd for landing
        AR (float): Aspect ratio
        e (float): oswald efficiency factor
        weight_ratio (float): Weight ratio
        wing_loading (Any): Wing loading

    Returns:
        tuple: Wing Loading, Thrust loading
    """
    N = no_of_engines

    g1: float = get_climb_rate_failed_approach(N)
    l_o_d: float = cl_app / (cd + cl_app**2 / (np.pi * AR * e))
    thrust_loading: float = N / (N - 1) * (g1 + 1 / l_o_d)
    thrust_loading_arr: FloatArray = shape_like(thrust_loading, wing_loading)
    return (wing_loading, thrust_loading_arr)
