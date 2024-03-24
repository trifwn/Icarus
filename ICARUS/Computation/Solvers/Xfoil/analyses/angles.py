from multiprocessing import Pool
from typing import Any

import numpy as np
from tqdm.auto import tqdm
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil

from ICARUS import CPU_TO_USE
from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Computation.Solvers.Xfoil.post_process.polars import save_multiple_reyn
from ICARUS.Core.types import FloatArray


def single_reynolds_run(
    Reyn: float,
    MACH: float,
    AoAmin: float,
    AoAmax: float,
    AoAstep: float,
    airfoil: Airfoil,
    solver_options: dict[str, Any] = {},
) -> FloatArray:
    xf = XFoil()
    xf.Re = Reyn
    xf.M = 0.0

    pts = airfoil.selig
    xpts = pts[0]
    ypts = pts[1]
    xf_airf_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = xf_airf_obj
    # xf.print = True
    for key, value in solver_options.items():
        if key == "repanel_n":
            print(f"Repaneling Airfoil with {value}")
            xf.repanel(value)
        else:
            setattr(xf, key, value)

    # xf.filter()
    # xf.max_iter = 500

    # If the values are both negative and positive split the into 2 run
    # if AoAmin < 0 and AoAmax > 0:
    #     aXF1, clXF1, cdXF1, cmXF1, cpXF1 = xf.aseq(0, AoAmin, -AoAstep)
    #     aXF2, clXF2, cdXF2, cmXF2, cpXF2 = xf.aseq(0, AoAmax, AoAstep)
    #     aXF = np.concatenate((aXF1, aXF2[1:]))
    #     clXF = np.concatenate((clXF1, clXF2[1:]))
    #     cdXF = np.concatenate((cdXF1, cdXF2[1:]))
    #     cmXF = np.concatenate((cmXF1, cmXF2[1:]))
    #     cpXF = np.concatenate((cpXF1, cpXF2[1:]))
    # else:
    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)
    df = np.array([aXF, clXF, cdXF, cmXF], dtype=float).T
    return df


def single_reynolds_run_seq(
    Reyn: float,
    MACH: float,
    angles: list[float] | FloatArray,
    airfoil: Airfoil,
    solver_options: dict[str, Any] = {},
) -> FloatArray:
    xf = XFoil()
    xf.Re = Reyn

    xpts, ypts = airfoil.selig
    airfoil_obj = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = airfoil_obj

    for key, value in solver_options.items():
        if key == "repanel_n":
            print(f"Repaneling Airfoil with {value}")
            xf.repanel(value)
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


def single_reynolds_star_run(args: Any) -> FloatArray:
    return single_reynolds_run(*args)


def single_reynolds_star_run_seq(args: Any) -> FloatArray:
    return single_reynolds_run_seq(*args)


def multiple_reynolds_serial(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    solver_options: dict[str, Any],
) -> None:
    data: list[FloatArray] = []

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

    save_multiple_reyn(airfoil, reyn_dicts, reynolds)


def multiple_reynolds_parallel(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    min_aoa: float,
    max_aoa: float,
    aoa_step: float,
    solver_options: dict[str, Any],
) -> None:
    data: list[FloatArray] = []

    with Pool(processes=CPU_TO_USE) as pool:
        args_list = [
            (reyn, mach, min_aoa, max_aoa, aoa_step, airfoil, solver_options)
            for reyn in reynolds
        ]
        data = list(
            tqdm(
                pool.imap(single_reynolds_star_run, args_list),
                total=len(reynolds),
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

    save_multiple_reyn(airfoil, reyn_dicts, reynolds)


def multiple_reynolds_parallel_seq(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, Any],
) -> None:
    data: list[FloatArray] = []

    with Pool(processes=CPU_TO_USE) as pool:
        args_list = [(reyn, mach, angles, airfoil, solver_options) for reyn in reynolds]
        data = list(
            tqdm(
                pool.imap(single_reynolds_star_run_seq, args_list),
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

    save_multiple_reyn(airfoil, reyn_dicts, reynolds)


def multiple_reynolds_serial_seq(
    airfoil: Airfoil,
    reynolds: list[float],
    mach: float,
    angles: list[float],
    solver_options: dict[str, Any],
) -> None:
    data: list[FloatArray] = []

    with tqdm(total=len(reynolds), colour="#FF0000") as t:
        for reyn in reynolds:
            t.desc = f"Reynolds: {reyn}:"
            clcdcm_xf = single_reynolds_run_seq(
                reyn,
                mach,
                angles,
                airfoil,
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

    save_multiple_reyn(airfoil, reyn_dicts, reynolds)
