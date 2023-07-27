from ast import arg
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from tqdm.auto import tqdm
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.types import FloatArray
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
) -> ndarray[Any, dtype[floating[Any]]]:

    xf = XFoil()
    xf.Re = Reyn

    for key, value in solver_options.items():
        setattr(xf, key, value)

    xpts, ypts = airfoil.selig
    naca = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca

    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)

    return np.array([aXF, clXF, cdXF, cmXF], dtype=float).T


def single_reynolds_star_run(args: Any) -> ndarray[Any, dtype[floating[Any]]]:
    return single_reynolds_run(*args)


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

    with tqdm(total=len(reynolds), colour="#FF0000") as t:
        for reyn in reynolds:
            t.desc = f"Reynolds: {reyn}:"
            clcdcm_xf = single_reynolds_run(
                Reyn=reyn,
                MACH=mach,
                AoAmin=min_aoa,
                AoAmax=max_aoa,
                AoAstep=aoa_step,
                airfoil=airfoil,
                solver_options=solver_options,
            )
            data.append(clcdcm_xf)
            t.update()

    reyn_dicts: list[dict[str, FloatArray]] = []
    for batchRe in data:
        tempDict: dict[str, FloatArray] = {}
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

    from multiprocessing import Pool, RLock, freeze_support

    with Pool(processes=6) as pool:
        args_list = [
            (reyn, mach, min_aoa, max_aoa, aoa_step, airfoil, solver_options)
            for reyn in reynolds
        ]
        data = list(
            tqdm(
                pool.imap(single_reynolds_star_run, args_list),
                total=len(args_list),
                bar_format="\t\tTotal Progres {l_bar}{bar:30}{r_bar}",
                position=0,
                leave=True,
                colour="#FF0000",
            ),
        )

    reyn_dicts: list[dict[str, FloatArray]] = []
    for batchRe in data:
        tempDict: dict[str, FloatArray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        reyn_dicts.append(tempDict)

    save_multiple_reyn(db.foilsDB, airfoil, reyn_dicts, reynolds)
