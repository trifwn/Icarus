from ICARUS.Software.GenuVP3.getWakeData import getWakeData
from ICARUS.Database.Database_3D import ang2case
import numpy as np


def gnvpGeom():

    from tests.planes import ap

    case = ang2case(5.)
    _, _, gridGNVP = getWakeData(ap, case)
    gridAP = []
    for surface in ap.surfaces:
        gridAP.append(surface.grid)
    gridAP = np.array(gridAP)
    shape = gridAP.shape
    gridAP = gridAP.reshape(shape[0]*shape[1]*shape[2], shape[3]) - ap.CG
    gridAP = np.meshgrid(gridAP)
    gridGNVP = np.meshgrid(gridGNVP)
    return gridAP, gridGNVP
