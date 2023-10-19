from typing import Any

# USEFULL LOAD

def usefull_load_criterion(
    thrust_loading: Any,
    wf_wg: float,
    w_p: float,
):
    """
    Returns the useful load fraction for a given thrust loading

    Args:
        thrust_loading (float): Thrust loading

    Returns:
        _type_: Useful load fraction
    """
    u_parameter = 0.76875 - 125/120 * (thrust_loading) 
    we_wg = 1 - u_parameter
    wg = w_p/(1-we_wg - wf_wg )

    # print(f"{w_p=}, {wg*wf_wg=}, {wg*we_wg=}")
    return wg
