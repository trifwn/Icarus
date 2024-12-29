from typing import Any

import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.computation.solvers.Xfoil.utils import angles_sepatation
from ICARUS.core.types import FloatArray


def return_cps(
    Reyn: float,
    MACH: float,
    angles: list[float],
    pts: FloatArray,
    ftrip_low: float = 1.0,
    ftrip_up: float = 1.0,
    Ncrit: float = 9,
) -> tuple[list[Any], list[Any], FloatArray]:
    """!TO BE DEPRECATED! SHOULD BE LOGGED AUTOMATICALLY AND NOT RETURNED

    Args:
        Reyn (float): _description_
        MACH (float): _description_
        angles (list[float]): _description_
        pts (FloatArray): _description_
        ftrip_low (float, optional): _description_. Defaults to 1.0.
        ftrip_up (float, optional): _description_. Defaults to 1.0.
        Ncrit (float, optional): _description_. Defaults to 9.

    Returns:
        _type_: _description_

    """
    xf = XFoil()
    xf.Re = Reyn
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
