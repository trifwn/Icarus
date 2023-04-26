import numpy as np

from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg
import ICARUS.Vehicle.wing as wing

def hermes(airfoils,name):
    Origin = np.array([0., 0., 0.])

    wingPos = np.array([0.0 - 0.159/4, 0.0, 0.0])
    wingOrientation = np.array([2.8, 0.0, 0.0])

    mainWing = wg(name="wing",
                airfoil=airfoils['NACA4415'],
                Origin=Origin + wingPos,
                Orientation=wingOrientation,
                isSymmetric=True,
                span=2 * 1.130,
                sweepOffset=0,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord=[0.159, 0.072],
                spanFun=wing.linSpan,
                N=30,
                M=5,
                mass=0.670)
    # mainWing.plotWing()
    
    elevatorPos = np.array([0.54 - 0.130/4, 0., 0.])
    elevatorOrientantion = np.array([0., 0., 0.])

    elevator = wg(name="tail",
                airfoil=airfoils['NACA0008'],
                Origin=Origin + elevatorPos,
                Orientation=elevatorOrientantion,
                isSymmetric=True,
                span=2 * 0.169,
                sweepOffset=0,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord=[0.130, 0.03],
                spanFun=wing.linSpan,
                N=15,
                M=5,
                mass=0.06)
    # elevator.plotWing()
    
    rudderPos = np.array([0.47 - 0.159/4, 0., 0.01])
    rudderOrientantion = np.array([0.0, 0.0, 90.0])

    rudder = wg(name="rudder",
                airfoil=airfoils['NACA0008'],
                Origin=Origin + rudderPos,
                Orientation=rudderOrientantion,
                isSymmetric=False,
                span=0.160,
                sweepOffset=0.1,
                dihAngle=0,
                chordFun=wing.linearChord,
                chord=[0.2, 0.1],
                spanFun=wing.linSpan,
                N=15,
                M=5,
                mass=0.04)
    # rudder.plotWing()
    
    liftingSurfaces = [mainWing, elevator, rudder]

    addedMasses = [
        (0.500 , np.array([-0.40, 0.0, 0.0])), # Motor
        (1.000 , np.array([0.090, 0.0, 0.0])), # Battery
        (0.900 , np.array([0.130, 0.0, 0.0])), # Payload
        ]
    ap = Plane(name, liftingSurfaces)

    # from ICARUS.Database import DB3D
    # ap.accessDB(HOMEDIR, DB3D)
    # ap.visAirplane()
    ap.addMasses(addedMasses)
    
    return ap