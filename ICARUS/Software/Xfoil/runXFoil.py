import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil


def anglesSep(anglesALL):
    pangles = []
    nangles = []
    for ang in anglesALL:
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
            nangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles


def runXFoil(
    Reyn,
    MACH,
    AoAmin,
    AoAmax,
    AoAstep,
    pts,
    ftrip_low=0.1,
    ftrip_up=0.1,
    Ncrit=9,
):
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


def batchRUN(Reynolds, MACH, AoAmin, AoAmax, AoAstep, pts):
    Data = []
    for Re in Reynolds:
        clcdcmXF = runXFoil(Re, MACH, AoAmin, AoAmax, AoAstep, pts)
        Data.append(clcdcmXF)

    Redicts = []
    for i, batchRe in enumerate(Data):
        tempDict = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        Redicts.append(tempDict)
    return Redicts


def saveXfoil(airfoils, polars, Reynolds):
    masterDir = os.getcwd()
    os.chdir(masterDir)
    for airf, clcdData in zip(airfoils, polars):
        os.chdir(masterDir)
        os.chdir(os.path.join("Data", "2D", f"NACA{airf}"))
        airfoilPath = os.getcwd()

        for i, ReynDat in enumerate(clcdData):
            os.chdir(airfoilPath)

            reyndir = f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}"
            os.makedirs(reyndir, exist_ok=True)
            os.chdir(reyndir)
            cwd = os.getcwd()

            for angle in ReynDat.keys():
                os.chdir(cwd)
                if float(angle) >= 0:
                    folder = str(angle)[::-1].zfill(7)[::-1]
                else:
                    folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
                os.makedirs(folder, exist_ok=True)
                os.chdir(folder)
                fname = "clcd.xfoil"
                with open(fname, "w") as file:
                    pols = angle
                    for i in ReynDat[angle]:
                        pols = pols + "\t" + str(i)
                    file.writelines(pols)


def plotBatch(data, Reynolds):
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


def returnCPs(Reyn, MACH, angles, pts, ftrip_low=1, ftrip_up=1, Ncrit=9):
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

    nangles, pangles = anglesSep(angles)
    cps = []
    cpsn = []
    x = []

    for a in pangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cps.append(cp)

    for a in nangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cpsn.append(cp)

    return [cpsn, nangles], [cps, pangles], x


def runAndSave(
    CASEDIR,
    HOMEDIR,
    Reyn,
    MACH,
    AoAmin,
    AoAmax,
    AoAstep,
    pts,
    ftrip_low=0.1,
    ftrip_up=0.2,
    Ncrit=9,
):
    os.chdir(CASEDIR)

    xf = XFoil()
    xf.Re = Reyn
    xf.n_crit = Ncrit
    # xf.M = MACH
    xf.max_iter = 100
    xf.print = False
    xpts, ypts = pts.T
    naca = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca

    if AoAmin * AoAmax < 0:
        max_tr = max(ftrip_low, ftrip_up)
        slope_up = (ftrip_up - max_tr) / (AoAmax)
        slope_low = (ftrip_low - max_tr) / (AoAmax)

        aXF1 = []
        clXF1 = []
        cdXF1 = []
        cmXF1 = []
        flag = 0
        for angle in np.arange(0, AoAmax + 0.5, 0.5):
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

        slope_up = (ftrip_low - max_tr) / (AoAmax)
        slope_low = (ftrip_up - max_tr) / (AoAmax)

        aXF2 = []
        clXF2 = []
        cdXF2 = []
        cmXF2 = []
        flag = 0
        for angle in np.arange(AoAmin, 0.5, 0.5)[::-1]:
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
        aXF, clXF, cdXF, cmXF, _ = xf.aseq(AoAmin, AoAmax, AoAstep)

    Res = np.vstack((aXF, clXF, cdXF, cmXF)).T
    df = pd.DataFrame(Res, columns=["AoA", "CL", "CD", "Cm"]).dropna(thresh=2)
    df = df.sort_values("AoA")
    df.to_csv("clcd.xfoil", index=False)
    os.chdir(HOMEDIR)
    return df
