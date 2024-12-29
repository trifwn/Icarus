from ICARUS.core.types import FloatArray


# FAR 3 -> 25.113
def far_3_takeoff(
    wing_loading: FloatArray,
    cl: float,
    l_t: float,
    sigma: float = 1,
) -> tuple[FloatArray, FloatArray]:
    """Returns the thrust loading for a given wing loading, cl_max and takeoff distance

    Args:
        wing_loading (float): Wing loading
        cl_max (float): Max lift coefficient
        L_t (float): Takeoff distance

    Returns:
        tuple: Wing Loading, Thrust loading

    """
    thrust_loading = 37.5 * wing_loading / (sigma * cl * l_t)

    return (wing_loading, thrust_loading)
