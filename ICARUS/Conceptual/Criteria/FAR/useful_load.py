from typing import Any

from ICARUS.Core.types import FloatArray

# USEFULL LOAD


def usefull_load_criterion(
    thrust_loading: float,
    wf_wg: float,
    w_p: float,
) -> float:
    """
    Returns the useful load fraction for a given thrust loading

    Args:
        thrust_loading (float): Thrust loading

    Returns:
        _type_: Useful load fraction
    """
    u_parameter = 0.76875 - 125 / 120 * (thrust_loading)
    we_wg = 1 - u_parameter
    wg = w_p / (1 - we_wg - wf_wg)

    return wg
