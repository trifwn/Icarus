from typing import Any

import numpy as np


# HELPER FUNCTIONS
def shape_like(in1: Any, in2: Any) -> Any:
    if type(in1) is type(in2):
        return in1
    if isinstance(in1, float) and isinstance(in2, np.ndarray):
        return np.array([in1] * len(in2))
    raise (ValueError("Invalid input types"))


def lift_drag_min_drag(cl: float, cd_0: float, AR: float, oswald: float) -> float:
    return float(np.sqrt(cl * np.pi * oswald * AR) / (2 * cd_0))


def get_climb_rate(no_of_engines: int) -> float:
    if no_of_engines == 4:
        return 3 / 100
    if no_of_engines == 3:
        return 2.7 / 100
    if no_of_engines == 2:
        return 2.4 / 100
    raise ValueError("Invalid number of engines")


def get_climb_rate_failed_approach(no_of_engines: int) -> float:
    if no_of_engines == 4:
        return 2.7 / 100
    if no_of_engines == 3:
        return 2.4 / 100
    if no_of_engines == 2:
        return 2.1 / 100
    raise ValueError("Invalid number of engines")


def drag_coeff_skin(
    cd_0: float,
    flap_extension: float,  # degrees
    landing_gear_cd: float = 0,
) -> float:
    # cd_i = cl**2/(np.pi*AR*oswald)
    if flap_extension > 10:
        cd_f = -0.005 + 0.001 * flap_extension
    else:
        cd_f = 0
    cd_l = landing_gear_cd
    return cd_0 + cd_f + cd_l
