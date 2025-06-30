from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.solvers.Xfoil.post_process.polars import save_multiple_reyn

if TYPE_CHECKING:
    from ICARUS.solvers.Xfoil.xfoil import XfoilSolverParameters


def single_reynolds_run(
    Reyn: float,
    MACH: float,
    AoAmin: float,
    AoAmax: float,
    AoAstep: float,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> FloatArray:
    xf = XFoil()
    xf.print = solver_parameters.print
    xf.Re = Reyn
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

    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)
    df = np.array([aXF, clXF, cdXF, cmXF], dtype=float).T
    return df


def single_reynolds_run_seq(
    Reyn: float,
    MACH: float,
    angles: list[float] | FloatArray,
    airfoil: Airfoil,
    solver_parameters: XfoilSolverParameters,
) -> FloatArray:
    xf = XFoil()
    xf.Re = Reyn
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
        clXF.append(cl)
        cdXF.append(cd)
        cmXF.append(cm)
        cpXF.append(cp)
        xf.reset_bls()
    return np.array([aXF, clXF, cdXF, cmXF], dtype=float).T


def multiple_reynolds_serial(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    solver_parameters: XfoilSolverParameters,
) -> None:
    data: list[FloatArray] = []

    for reyn in reynolds:
        clcdcm_xf = single_reynolds_run(
            Reyn=reyn,
            MACH=mach,
            AoAmin=min_aoa,
            AoAmax=max_aoa,
            AoAstep=aoa_step,
            airfoil=airfoil,
            solver_parameters=solver_parameters,
        )
        data.append(clcdcm_xf)

    reyn_dicts: list[dict[str, FloatArray]] = []
    for batchRe in data:
        tempDict: dict[str, FloatArray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        reyn_dicts.append(tempDict)
    save_multiple_reyn(airfoil, reyn_dicts, reynolds)


def multiple_reynolds_serial_seq(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_parameters: XfoilSolverParameters,
) -> None:
    data: list[FloatArray] = []

    for reyn in reynolds:
        clcdcm_xf = single_reynolds_run_seq(
            reyn,
            mach,
            angles,
            airfoil,
            solver_parameters=solver_parameters,
        )
        data.append(clcdcm_xf)

    reyn_dicts: list[dict[str, FloatArray]] = []
    for batchRe in data:
        tempDict: dict[str, FloatArray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        reyn_dicts.append(tempDict)

    save_multiple_reyn(airfoil, reyn_dicts, reynolds)
