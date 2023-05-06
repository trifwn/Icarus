import numpy as np

from ICARUS.Database.utils import ang2case
from ICARUS.Software.GenuVP3.postProcess.getWakeData import getWakeData
from ICARUS.Visualization.GNVPwake import GNVPwake


def gnvpGeom():

    from Data.Planes.simple_wing import airplane

    case = ang2case(2.0)
    _, _, gridGNVP = getWakeData(airplane, case)
    gridAP = []
    for surface in airplane.surfaces:
        gridAP.append(surface.getGrid())
    gridAP = np.array(gridAP)
    shape = gridAP.shape
    gridAP = gridAP.reshape(shape[0] * shape[1] * shape[2], shape[3]) - airplane.CG
    gridAP = np.meshgrid(gridAP)
    gridGNVP = np.meshgrid(gridGNVP)
    GNVPwake(airplane, case)
    return gridAP, gridGNVP
