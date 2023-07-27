from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.Software.Xfoil.utils import angles_sepatation


def return_cps(
    Reyn: float,
    MACH: float,
    angles: list[float],
    pts: ndarray[Any, dtype[floating[Any]]],
    ftrip_low: float = 1.0,
    ftrip_up: float = 1.0,
    Ncrit: float = 9,
) -> tuple[list[Any], list[Any], ndarray[Any, dtype[floating[Any]]]]:
    """
    !TO BE DEPRECATED! SHOULD BE LOGGED AUTOMATICALLY AND NOT RETURNED

    Args:
        Reyn (float): _description_
        MACH (float): _description_
        angles (list[float]): _description_
        pts (ndarray[Any, dtype[floating[Any]]]): _description_
        ftrip_low (float, optional): _description_. Defaults to 1.0.
        ftrip_up (float, optional): _description_. Defaults to 1.0.
        Ncrit (float, optional): _description_. Defaults to 9.

    Returns:
        _type_: _description_
    """
    xf = XFoil()
    xf.Re = Reyn
    print(MACH)
    # xf.M = MACH
    xf.n_crit = Ncrit
    xf.xtr = (ftrip_low, ftrip_up)
    xf.max_iter = 400
    xf.print = False
    xpts, ypts = pts.T
    airf = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = airf
    # AoAmin = min(angles)
    # AoAmax = max(angles)

    nangles, pangles = angles_sepatation(angles)
    cps = []
    cpsn = []
    x = np.array([], dtype=float)
    for a in pangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cps.append(cp)

    for a in nangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cpsn.append(cp)

    return [cpsn, nangles], [cps, pangles], x
