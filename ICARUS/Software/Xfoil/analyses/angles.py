import numpy as np
from numpy import dtype, floating, ndarray
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil


from typing import Any
from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database.db import DB

from ICARUS.Software.Xfoil.post_process.polars import save_multiple_reyn


def single_reynolds_run(
    Reyn: float,
    MACH: float,
    AoAmin: float,
    AoAmax: float,
    AoAstep: float,
    airfoil: AirfoilD,
    solver_options: dict[str, Any] = {},
)-> ndarray[Any, dtype[floating[Any]]]:

    xf = XFoil()
    xf.Re = Reyn

    for key, value in solver_options.items():
        setattr(xf, key, value)
        

    xpts, ypts = airfoil.selig
    naca = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca
    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)

    return np.array([aXF, clXF, cdXF, cmXF], dtype = float).T


def multiple_reynolds_serial(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    solver_options: dict[str, Any],
) -> None:

    data: list[ndarray[Any, dtype[floating[Any]]]] = []
    for reyn in reynolds:
        clcdcm_xf = single_reynolds_run(reyn, mach, min_aoa, max_aoa, aoa_step, airfoil, solver_options)
        data.append(clcdcm_xf)

    reyn_dicts: list[dict[str, ndarray]] = []
    for i, batchRe in enumerate(data):
        tempDict: dict[str, ndarray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        reyn_dicts.append(tempDict)

    save_multiple_reyn(db.foilsDB, airfoil, reyn_dicts, reynolds)

def multiple_reynolds_parallel(
    db: DB,
    airfoil: AirfoilD,
    reynolds: list[float],
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    solver_options: dict[str, Any],
) -> None:

    data: list[ndarray[Any, dtype[floating[Any]]]] = []

    from multiprocessing import Pool
    with Pool(12) as pool:
        args_list = [
            (
                reyn,
                mach,
                min_aoa,
                max_aoa,
                aoa_step,
                airfoil,
                solver_options
            )
            for reyn in reynolds
        ]
        data = pool.starmap(single_reynolds_run, args_list)

    reyn_dicts: list[dict[str, ndarray]] = []
    for i, batchRe in enumerate(data):
        tempDict: dict[str, ndarray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        reyn_dicts.append(tempDict)

    save_multiple_reyn(db.foilsDB, airfoil, reyn_dicts, reynolds)
