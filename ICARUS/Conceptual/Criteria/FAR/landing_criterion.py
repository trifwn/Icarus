import numpy as np
from typing import Any
from .helper_functions import shape_like
# FAR 1 -> 25.111
def far_1_landing(
    l_landing: float,
    cl_approach: float, 
    thrust_loading: Any,
    sigma: float =1,
):
    """
    Returns the wing loading for a given cl_max and vstall

    Args:
        l_landing (float): Landing Distance
        cl_approach (float): CL at approach
        weight_ratio (float): Weight ratio
        thrust_loading (Any): Thrust loading
        sigma (float, optional): Desnity Fraction . Defaults to 1.

    Returns:
        _type_: (wing_loading, thrust_loading)
    """
    
    wing_loading = l_landing* sigma * cl_approach / (0.3 * 17.15**2)
    wing_loading = shape_like(wing_loading,thrust_loading) 
    return (wing_loading, thrust_loading)


def far_inverse_landing_criterion_cl_max(
    V_app: float,
    wing_loading: float,
    sigma = 1,
) -> float:
    cl_app = (17.15/V_app) **2 * (wing_loading /sigma)
    landing_dist = 0.3 * V_app**2
    cl_max = cl_app * 1.69
    print(f"{landing_dist=} m, {V_app=} m/s, {cl_max=}")
    return cl_app * 1.69

def far_inverse_landing_criterion_cl_max2(
    l_landing: float,
    wing_loading: float,
    sigma = 1,
):
    V_app = np.sqrt(l_landing / 0.3)
    cl_app = (17.15/V_app) **2 * (wing_loading /sigma)
    print(f"{l_landing=} m, {V_app=} m/s, {1.69 * cl_app=}")
    return cl_app * 1.69    