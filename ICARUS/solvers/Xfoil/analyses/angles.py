from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray

if TYPE_CHECKING:
    from ICARUS.solvers.Xfoil.xfoil import XfoilSolverParameters


def aseq_analysis(
    reynolds: float,
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> FloatArray:
    xf = XFoil()
    xf.print = solver_parameters.print
    xf.Re = reynolds
    xf.M = 0.0

    pts = airfoil.selig
    xpts = pts[0]
    ypts = pts[1]
    xf_airf_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = xf_airf_obj

    params_dict = asdict(solver_parameters)
    for key, value in params_dict.items():
        if key == "repanel_n":
            if value > 0:
                print(f"Repaneling Airfoil with {value}")
                xf.repanel(value)
        elif key == "print":
            continue
        else:
            setattr(xf, key, value)

    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(min_aoa, max_aoa, aoa_step)
    df = np.array([aXF, clXF, cdXF, cmXF], dtype=float).T
    return df


def aseq_analysis_reset_bl(
    reynolds: float,
    mach: float,
    angles: list[float] | FloatArray,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> FloatArray:
    xf = XFoil()
    xf.Re = reynolds
    xf.M = 0.0

    xpts, ypts = airfoil.selig
    airfoil_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = airfoil_obj

    params_dict = asdict(solver_parameters)
    for key, value in params_dict.items():
        if key == "repanel_n":
            if value > 0:
                print(f"Repaneling Airfoil with {value}")
                xf.repanel(value)
        elif key == "print":
            xf.print = value
        else:
            setattr(xf, key, value)

    # xf.filter()

    aXF = []
    clXF = []
    cdXF = []
    cmXF = []
    cpXF = []

    for angle in angles:
        aXF.append(angle)
        cl, cd, cm, cp = xf.a(angle)
        # x, y, cp = xf.get_cp_distribution()

        clXF.append(cl)
        cdXF.append(cd)
        cmXF.append(cm)
        cpXF.append(cp)
        xf.reset_bls()
    return np.array([aXF, clXF, cdXF, cmXF], dtype=float).T
