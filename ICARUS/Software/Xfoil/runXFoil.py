import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil


def angles_sepatation(all_angles: list[float]) -> tuple[list[float], list[float]]:
    """Separate angles in positive and negative.

    Args:
        anglesALL (_type_): _description_

    Returns:
        _type_: _description_
    """

    pangles: list[float] = []
    nangles: list[float] = []
    for ang in all_angles:
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
            nangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles


def run_xfoil(
    Reyn: float,
    MACH: float,
    AoAmin: float,
    AoAmax: float,
    AoAstep: float,
    pts: ndarray[Any, dtype[floating]],
    ftrip_low: float = 0.1,
    ftrip_up: float = 0.1,
    Ncrit: float = 9,
) -> ndarray[Any, dtype[floating]]:
    xf = XFoil()
    xf.Re = Reyn
    xf.n_crit = Ncrit
    # xf.M = MACH
    xf.xtr = (ftrip_low, ftrip_up)
    xf.max_iter = 100
    xf.print = False
    xpts, ypts = pts.T
    naca = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca
    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)
    return np.array([aXF, clXF, cdXF, cmXF]).T


def multiple_reynolds_run(
    Reynolds: list[float],
    MACH: float,
    AoAmin: float,
    AoAmax: float,
    AoAstep: float,
    pts: ndarray[Any, dtype[floating]],
) -> list[dict[str, ndarray]]:

    Data: list[ndarray[Any, dtype[floating]]] = []
    for Re in Reynolds:
        clcdcmXF = run_xfoil(Re, MACH, AoAmin, AoAmax, AoAstep, pts)
        Data.append(clcdcmXF)

    Redicts: list[dict[str, ndarray]] = []
    for i, batchRe in enumerate(Data):
        tempDict: dict[str, ndarray] = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        Redicts.append(tempDict)
    return Redicts


def saveXfoil(
    airfoils: str,
    polars: list[list[dict[str, ndarray]]],
    Reynolds: list[float],
) -> None:

    masterDir: str = os.getcwd()
    os.chdir(masterDir)
    for airfoil, clcdData in zip(airfoils, polars):
        os.chdir(masterDir)
        os.chdir(os.path.join("Data", "2D", f"NACA{airfoil}"))
        airfoilPath: str = os.getcwd()

        for i, ReynDat in enumerate(clcdData):
            os.chdir(airfoilPath)

            reyndir: str = f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}"
            os.makedirs(reyndir, exist_ok=True)
            os.chdir(reyndir)
            cwd: str = os.getcwd()

            for angle in ReynDat.keys():
                os.chdir(cwd)
                if float(angle) >= 0:
                    folder: str = str(angle)[::-1].zfill(7)[::-1]
                else:
                    folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
                os.makedirs(folder, exist_ok=True)
                os.chdir(folder)
                fname = "clcd.xfoil"
                with open(fname, "w") as file:
                    pols = str(angle)
                    for i in ReynDat[angle]:
                        pols += f"\t{str(i)}"
                    file.writelines(pols)


def plotBatch(data: list[list[dict[str, ndarray]]], Reynolds: list[float]) -> None:
    for airfpol in data:
        for i, dict1 in enumerate(airfpol):
            a = np.vstack(
                [
                    np.hstack((float(key), np.float64(dict1[key])))
                    for key in dict1.keys()
                ],
            ).T
            plt.plot(
                a[0, :],
                a[1, :],
                label=f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}",
            )
    plt.legend()


def returnCPs(
    Reyn: float,
    MACH: float,
    angles: list[float],
    pts: ndarray[Any, dtype[floating]],
    ftrip_low: float = 1.0,
    ftrip_up: float = 1.0,
    Ncrit: float = 9,
) -> tuple[list[Any], list[Any], ndarray[Any, dtype[floating]]]:
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
    x = []

    for a in pangles:
        xf.a(a)
        x, cp = xf.get_cp_distribution()
        cps.append(cp)

    for a in nangles:
        xf.a(a)
        x, cp = xf.get_cp_distribution()
        cpsn.append(cp)

    return [cpsn, nangles], [cps, pangles], x


def run_and_save(
    CASEDIR: str,
    HOMEDIR: str,
    reynolds: float,
    MACH: float,
    aoa_min: float,
    aoa_max: float,
    aoa_step: float,
    pts: ndarray[Any, dtype[floating]],
    ftrip_low: float = 0.1,
    ftrip_up: float = 0.2,
    n_crit: float = 9,
) -> DataFrame:
    os.chdir(CASEDIR)

    xf = XFoil()
    xf.Re = reynolds
    xf.n_crit = n_crit
    # xf.M = MACH
    xf.max_iter = 100
    xf.print = False
    xpts, ypts = pts.T
    naca = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca

    if aoa_min * aoa_max < 0:
        max_tr = max(ftrip_low, ftrip_up)
        slope_up = (ftrip_up - max_tr) / aoa_max
        slope_low = (ftrip_low - max_tr) / aoa_max

        aXF1: ndarray[Any, dtype[floating]] = np.array([])
        clXF1: ndarray[Any, dtype[floating]] = np.array([])
        cdXF1: ndarray[Any, dtype[floating]] = np.array([])
        cmXF1: ndarray[Any, dtype[floating]] = np.array([])
        flag = 0
        for angle in np.arange(0, aoa_max + 0.5, 0.5):
            f_up = max_tr + slope_up * angle
            f_low = max_tr + slope_low * angle
            xf.xtr = (f_up, f_low)

            cl, cd, cm, _ = xf.a(angle)
            if np.isnan(cl):
                flag += 1
            if flag > 2:
                break
            aXF1 = np.hstack((aXF1, angle))
            clXF1 = np.hstack((clXF1, cl))
            cdXF1 = np.hstack((cdXF1, cd))
            cmXF1 = np.hstack((cmXF1, cm))

        xf.reset_bls()

        slope_up = (ftrip_low - max_tr) / aoa_max
        slope_low = (ftrip_up - max_tr) / aoa_max

        aXF2: ndarray[Any, dtype[floating]] = np.array([])
        clXF2: ndarray[Any, dtype[floating]] = np.array([])
        cdXF2: ndarray[Any, dtype[floating]] = np.array([])
        cmXF2: ndarray[Any, dtype[floating]] = np.array([])
        flag = 0
        for angle in np.arange(aoa_min, 0.5, 0.5)[::-1]:
            f_up = max_tr - slope_up * angle
            f_low = max_tr - slope_low * angle
            xf.xtr = (f_up, f_low)

            cl, cd, cm, _ = xf.a(angle)
            if np.isnan(cl):
                flag += 1
            if flag > 2:
                break

            aXF2 = np.hstack((aXF2, angle))
            clXF2 = np.hstack((clXF2, cl))
            cdXF2 = np.hstack((cdXF2, cd))
            cmXF2 = np.hstack((cmXF2, cm))

        aXF = np.hstack((aXF1, aXF2[1:]))
        clXF = np.hstack((clXF1, clXF2[1:]))
        cdXF = np.hstack((cdXF1, cdXF2[1:]))
        cmXF = np.hstack((cmXF1, cmXF2[1:]))
    else:
        xf.xtr = (ftrip_low, ftrip_up)
        aXF, clXF, cdXF, cmXF, _ = xf.aseq(aoa_min, aoa_max, aoa_step)

    Res = np.vstack((aXF, clXF, cdXF, cmXF)).T
    df: DataFrame = (
        DataFrame(Res, columns=["AoA", "CL", "CD", "Cm"])
        .dropna(thresh=2)
        .sort_values("AoA")
    )
    df.to_csv("clcd.xfoil", index=False)
    os.chdir(HOMEDIR)
    return df
