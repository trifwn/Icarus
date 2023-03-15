from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil
from ...Airfoils import airfoil as af
import numpy as np
import matplotlib.pyplot as plt


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
    Reyn, MACH, AoAmin, AoAmax, AoAstep, airfoil, ftrip_low=0.1, ftrip_up=0.2, Ncrit=9
):
    n_points = 100
    pts = af.saveAirfoil([[], [], airfoil, 0, n_points])
    xf = XFoil()
    xf.Re = Reyn
    xf.n_crit = Ncrit
    # xf.M = MACH
    xf.xtr = (ftrip_low, ftrip_up)
    xf.max_iter = 100
    xf.print = False
    xpts, ypts = pts.T
    naca0008 = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca0008
    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, AoAstep)
    return np.array([aXF, clXF, cdXF, cmXF]).T


def batchRUN(Reynolds, MACH, AoAmin, AoAmax, AoAstep, airfoil):
    Data = []
    for Re in Reynolds:
        clcdcmXF = runXFoil(Re, MACH, AoAmin, AoAmax, AoAstep, airfoil)
        Data.append(clcdcmXF)

    Redicts = []
    for i, batchRe in enumerate(Data):
        tempDict = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        Redicts.append(tempDict)
    return Redicts


def plotBatch(data, Reynolds):
    for airfpol in data:
        for i, dict1 in enumerate(airfpol):
            a = np.vstack([np.hstack((float(key), np.float64(dict1[key])))
                          for key in dict1.keys()]).T
            plt.plot(
                a[0, :], a[1, :], label=f"Reynolds_{np.format_float_scientific(Reynolds[i],sign=False,precision=3).replace('+', '')}")
    plt.legend()


def returnCPs(Reyn, MACH, angles, airfoil, ftrip_low=1, ftrip_up=1, Ncrit=9):
    n_points = 100
    pts = af.saveAirfoil([[], [], airfoil, 0, n_points])
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
