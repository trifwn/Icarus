from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


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
    Reyn, MACH, AoAmin, AoAmax, AoAstep, pts, ftrip_low=0.1, ftrip_up=0.1, Ncrit=9
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
    for airfoil, clcdData in zip(airfoils, polars):
        os.chdir(masterDir)
        os.chdir(f"Database/2D/NACA{airfoil}")
        airfoilPath = os.getcwd()

        for i, ReynDat in enumerate(clcdData):
            os.chdir(airfoilPath)

            reyndir = f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}"
            os.system(f"mkdir -p {reyndir}")
            os.chdir(reyndir)
            cwd = os.getcwd()

            for angle in ReynDat.keys():
                os.chdir(cwd)
                if float(angle) >= 0:
                    folder = str(angle)[::-1].zfill(7)[::-1] + "/"
                else:
                    folder = "m" + \
                        str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"
                os.system(f"mkdir -p {folder}")
                os.chdir(folder)
                fname = 'clcd.xfoil'
                with open(fname, 'w') as file:
                    pols = angle
                    for i in ReynDat[angle]:
                        pols = pols + "\t" + str(i)
                    file.writelines(pols)


def plotBatch(data, Reynolds):
    for airfpol in data:
        for i, dict1 in enumerate(airfpol):
            a = np.vstack([np.hstack((float(key), np.float64(dict1[key])))
                          for key in dict1.keys()]).T
            plt.plot(
                a[0, :], a[1, :], label=f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}")
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
    naca0008 = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca0008
    AoAmin = min(angles)
    AoAmax = max(angles)

    nangles, pangles = anglesSep(angles)
    cps = []
    cpsn = []

    for a in pangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cps.append(cp)

    for a in pangles:
        xf.a(a)
        x, y, cp = xf.get_cp_distribution()
        cpsn.append(cp)
    return [cpsn, nangles], [cps, pangles], x


def runAndSave(CASEDIR, HOMEDIR, Reyn, MACH, AoAmin, AoAmax, AoAstep, pts, ftrip_low=0.1, ftrip_up=0.1, Ncrit=9):
    os.chdir(CASEDIR)

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
    aXF1, clXF1, cdXF1, cmXF1, _ = xf.aseq(0, AoAmax, AoAstep)
    aXF2, clXF2, cdXF2, cmXF2, _ = xf.aseq(0, AoAmin, -AoAstep)

    aXF = np.hstack((aXF1, aXF2[1:]))
    clXF = np.hstack((clXF1, clXF2[1:]))
    cdXF = np.hstack((cdXF1, cdXF2[1:]))
    cmXF = np.hstack((cmXF1, cmXF2[1:]))

    Res = np.array([aXF, clXF, cdXF, cmXF], dtype=float).T
    df = pd.DataFrame(Res, columns=['AoA', 'CL', 'CD', 'Cm'])
    df = df.sort_values("AoA")
    df.to_csv('clcd.xfoil', index=False)
    os.chdir(HOMEDIR)
    return df
