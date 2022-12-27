from xfoil import XFoil
from xfoil.model import Airfoil as XFAirfoil
from . import airfoil as af
import numpy as np


def runXFoil(Reyn, MACH, angles, airfoil, ftrip_low=1, ftrip_up=1):
    n_points = 100
    pts = af.saveAirfoil([[], [], airfoil, 0, n_points])
    xf = XFoil()
    xf.Re = Reyn
    # xf.M = MACH
    xf.xtr = (ftrip_low, ftrip_up)
    xf.max_iter = 400
    xf.print = False
    xpts, ypts = pts.T
    naca0008 = XFAirfoil(x=xpts, y=ypts)
    xf.airfoil = naca0008
    AoAmin = min(angles)
    AoAmax = max(angles)
    aXF, clXF, cdXF, cmXF, cpXF = xf.aseq(AoAmin, AoAmax, 0.25)
    return np.array([aXF, clXF, cdXF, cmXF]).T
