import numpy as np

import ICARUS.Vehicle.wing as wing
from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg


def wing_var_chord_offset(airfoils, name:str, chords: list, offset: float):
    Origin = np.array([0., 0., 0.])
    wingPos = np.array([0.0 - 0.159/4, 0.0, 0.0])
    wingOrientation = np.array([2.8, 0.0, 0.0])

    mainWing = wg(name="wing",
                airfoil=airfoils['NACA4415'],
                Origin=Origin + wingPos,
                Orientation=wingOrientation,
                isSymmetric=True,
                span=2 * 1.130,
                sweepOffset= offset,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord= chords, #  [0.159, 0.072],
                spanFun=wing.linSpan,
                N=30,
                M=5,
                mass=0.670)
    # mainWing.plotWing()
    liftingSurfaces = [mainWing]

    addedMasses = [
        (0.500 , np.array([-0.40, 0.0, 0.0])), # Motor
        (1.000 , np.array([0.090, 0.0, 0.0])), # Battery
        (0.900 , np.array([0.130, 0.0, 0.0])), # Payload
        ]
    ap = Plane(name, liftingSurfaces)
    ap.addMasses(addedMasses)
    return ap