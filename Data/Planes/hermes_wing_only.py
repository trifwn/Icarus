import numpy as np

import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg


def hermesMainWing(airfoils,name: str) -> Plane:
    """Function to get a plane Consisting only of the main wing of the hermes plane

    Args:
        airfoils (_type_): _description_
        name (_type_): _description_

    Returns:
        _type_: _description_
    """
    origin = np.array([0., 0., 0.])

    wingPos = np.array([0.0, 0.0, 0.0])
    wingOrientation = np.array([2.8, 0.0, 0.0])

    mainWing = wg(name="wing",
                airfoil=airfoils['NACA4415'],
                Origin=origin + wingPos,
                Orientation=wingOrientation,
                isSymmetric=True,
                span=2 * 1.130,
                sweepOffset=0,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord=[0.159, 0.072],
                spanFun=wing.linSpan,
                N=20,
                M=5,
                mass=0.670)
    # mainWing.plotWing()

    liftingSurfaces = [mainWing]
    ap = Plane(name, liftingSurfaces)

    # ap.visAirplane()

    # addedMasses = [
    #     (0.500 , np.array([-0.40, 0.0, 0.0])), # Motor
    #     (1.000 , np.array([0.090, 0.0, 0.0])), # Battery
    #     (0.900 , np.array([0.130, 0.0, 0.0])), # Payload
    #     ]
    # ap.addMasses(addedMasses) 
    return ap