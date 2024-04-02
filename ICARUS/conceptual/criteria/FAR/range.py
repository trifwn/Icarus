import numpy as np

# RANGE


def range_criterion(
    range: float,
    mach: float,
    l_over_d: float,
    sfc: float,
) -> float:
    """
    Returns the Fuel fraction for a given range, mach, l_d_dmin, sfc and altitude

    Args:
        range (float): Range in nautical miles
        mach (float): Mach number
        l_d_dmin (float): Lift to drag ratio
        sfc (float): Specific fuel consumption
        altitude (float): Altitude

    Returns:
        _type_: Fuel fraction
    """
    a = 577  # speed of sound
    B: float = a * mach * l_over_d / sfc  # Breguet range
    wf_over_wg: float = 1 - 1 / np.exp(range / B)  # fuel fraction (wf/wg)
    return wf_over_wg
