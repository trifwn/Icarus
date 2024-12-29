import numpy as np

from ICARUS.core.types import FloatArray

from .helper_functions import shape_like


# FAR 1 -> 25.111
def far_1_landing(
    l_landing: float,
    cl_approach: float,
    thrust_loading: FloatArray,
    sigma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """Returns the wing loading for a given cl_max and vstall

    Args:
        l_landing (float): Landing Distance
        cl_approach (float): CL at approach
        weight_ratio (float): Weight ratio
        thrust_loading (Any): Thrust loading
        sigma (float, optional): Desnity Fraction . Defaults to 1.

    Returns:
        _type_: (wing_loading, thrust_loading)

    """
    wing_loading: float = l_landing * sigma * cl_approach / (0.3 * 17.15**2)
    wing_loading_arr: FloatArray = shape_like(wing_loading, thrust_loading)
    return (wing_loading_arr, thrust_loading)


def far_inverse_landing_criterion_cl_max(
    V_app: float,
    wing_loading: float,
    sigma: float = 1,
) -> float:
    cl_app: float = (17.15 / V_app) ** 2 * (wing_loading / sigma)
    # landing_dist: float = 0.3 * V_app**2
    # cl_max: float = cl_app * 1.69
    return cl_app * 1.69


def far_inverse_landing_criterion_cl_max2(
    l_landing: float,
    wing_loading: float,
    sigma: float = 1,
) -> float:
    V_app: float = np.sqrt(l_landing / 0.3)
    cl_app: float = (17.15 / V_app) ** 2 * (wing_loading / sigma)
    return float(cl_app * 1.69)
