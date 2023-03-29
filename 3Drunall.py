from Flight_Dynamics.dyn_plane import dyn_plane as dp
import numpy as np
import os

from Software.GenuVP import runGNVP as gnvp

from PlaneDefinition.plane import Airplane as Plane
from PlaneDefinition.wing import Wing as wg
import PlaneDefinition.wing as wing

from Database.getresults import Database_2D
from Database import DB3D, BASEGNVP
import time


start_time = time.time()
HOMEDIR = os.getcwd()

# # Airfoil Data
db = Database_2D(HOMEDIR)
airfoils = db.getAirfoils()
polars2D = db.Data


# # Get Plane
Origin = np.array([0., 0., 0.])

wingPos = np.array([0.0, 0.0, 0.0])
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
              N=20,
              M=13,
              mass=0.670)
# mainWing.plotWing()

elevatorPos = np.array([0.54, 0., 0.])
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
              M=8,
              mass=0.06)
# elevator.plotWing()

rudderPos = np.array([0.47, 0., 0.01])
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
            M=8,
            mass=0.04)
# rudder.plotWing()

liftingSurfaces = [mainWing, elevator, rudder]
addedMasses = [
    (0.500, np.array([-0.40, 0.0, 0.0])),  # Motor
    (1.000, np.array([0.090, 0.0, 0.0])),  # Battery
    (0.900, np.array([0.130, 0.0, 0.0])),  # Payload
]
ap = Plane("Hermes", liftingSurfaces)
# ap.visAirplane()

ap.accessDB(HOMEDIR, DB3D)
ap.addMasses(addedMasses)

cleaning = False
calcGenu = True
calcBatchGenu = True
petrubationAnalysis = True

# ## AoA Run
AoAmax = 10
AoAmin = -6
NoAoA = (AoAmax - AoAmin) + 1
angles = np.linspace(AoAmin, AoAmax, NoAoA)
Uinf = 20

if calcBatchGenu == True:
    genuBatchArgs = [ap, BASEGNVP, polars2D, "Xfoil", Uinf, angles]
    ap.runSolver(gnvp.runGNVPangles, genuBatchArgs)
genuPolarArgs = [ap.CASEDIR, HOMEDIR]
ap.makePolars(gnvp.makePolar, genuPolarArgs)
ap.defineSim(Uinf, 1.225)
ap.save()


# # Dynamics


# ### Define and Trim Plane
dyn = dp(ap, polars2D)
dyn.trim

# ### Pertrubations
dyn.allPerturb(5e-2, "Central")
dyn.get_pertrub()

if petrubationAnalysis == True:
    genuBatchArgs = [dyn, BASEGNVP, polars2D,
                     "Xfoil", dyn.trim['U'], dyn.trim['AoA']]
    dyn.accessDB(HOMEDIR)
    dyn.runAnalysis(gnvp.runGNVPpertr, genuBatchArgs)
genuLogArgs = [dyn.DynDir, HOMEDIR]
dyn.logResults(gnvp.logResults, genuLogArgs)
dyn.save()

dyn.scheme = "Central"
dyn.longitudalStability()
dyn.lateralStability()
dyn.save()

print(dyn.Along)
# [xu, xw, xq, xth]
# [zu, zw, zq, zth]
# [mu, mw, mq, mth]
# [0,  1 , 0 ,  0 ]

# dyn.AstarLong
print(dyn.Alat)
# [yv, yp , yr, yphi]
# [lv, lp , lr, lphi]
# [nv, n_p, nr, nph ]
# [0 , 1  , 0 , 0   ]
# dyn.AstarLat

print("PROGRAM EXECUTION TOOK --- %s seconds ---" % (time.time() - start_time))
