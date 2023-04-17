from ICARUS.Flight_Dynamics.dyn_plane import dyn_Airplane as dp
import numpy as np
import time
import os

from ICARUS.Vehicle.plane import Airplane as Plane
from ICARUS.Vehicle.wing import Wing as wg
import ICARUS.Vehicle.wing as wing

from ICARUS.Software.GenuVP3.angles import runGNVPangles, runGNVPanglesParallel
from ICARUS.Software.GenuVP3.pertrubations import runGNVPpertr, runGNVPpertrParallel
from ICARUS.Software.GenuVP3.pertrubations import runGNVPsensitivity, runGNVPsensitivityParallel
from ICARUS.Software.GenuVP3.filesInterface import makePolar, pertrResults
from ICARUS.Software.XFLR5.polars import readPolars2D

from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database import DB3D

start_time = time.time()
HOMEDIR = os.getcwd()

# # Airfoil Data
db = Database_2D(HOMEDIR)
airfoils = db.getAirfoils()
readPolars2D(db, HOMEDIR, f"{HOMEDIR}/ICARUS/Database/XFLR5/")
polars2D = db.Data


# # Get Plane
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
    (0.500, np.array([-0.40, 0.0, 0.0])),  # Motor
    (1.000, np.array([0.090, 0.0, 0.0])),  # Battery
    (0.900, np.array([0.130, 0.0, 0.0])),  # Payload
]
ap = Plane('HermesTTT', liftingSurfaces)
ap.visAirplane()

ap.accessDB(HOMEDIR, DB3D)
ap.addMasses(addedMasses)

calcPolarsGNVP3 = True
petrubationAnalysis = True
sensitivityAnalysis = False

# ## AoA Run
AoAmin = -6
AoAmax = 10
NoAoA = (AoAmax - AoAmin) + 1
angles = np.linspace(AoAmin, AoAmax, NoAoA)

Uinf = 20
ap.defineSim(Uinf, 1.225)
maxiter = 2
timestep = 0.1
ap.save()

if calcPolarsGNVP3 == True:
    polars_time = time.time()
    GNVP3BatchArgs = [ap, polars2D, "XFLR",
                      maxiter, timestep, Uinf, angles]
    ap.runAnalysis(runGNVPanglesParallel, GNVP3BatchArgs)

    print("Polars took : --- %s seconds --- in Parallel Mode" %
          (time.time() - polars_time))
GNVP3PolarArgs = [ap.CASEDIR, HOMEDIR]
ap.setPolars(makePolar, GNVP3PolarArgs)
ap.save()

# Dynamics
try:
    # ### Define and Trim Plane
    dyn = dp(ap)

    # ### Pertrubations
    dyn.allPerturb("Central")
    dyn.get_pertrub()

    if petrubationAnalysis == True:
        pert_time = time.time()
        GNVP3PertrArgs = [dyn, polars2D, "XFLR", maxiter, timestep,
                          dyn.trim['U'], dyn.trim['AoA']]
        dyn.accessDynamics(HOMEDIR)
        print("Running Pertrubations")
        dyn.runAnalysis(runGNVPpertrParallel, GNVP3PertrArgs)
        print("Pertrubations took : --- %s seconds ---" %
              (time.time() - pert_time))

    GNVP3LogArgs = [dyn.DynDir, HOMEDIR]
    dyn.setPertResults(pertrResults, GNVP3LogArgs)
    dyn.save()

    # SENSITIVITY ANALYSIS
    if sensitivityAnalysis == True:
        sens_time = time.time()
        for var in ['u', 'w', 'q', 'theta', 'v', 'p', 'r', 'phi']:
            space = np.logspace(np.log10(0.00001), np.log10(1), 10, base=10)
            space = [*-space, *space]
            maxiter = 2
            timestep = 5e-2
            dyn.sensitivityAnalysis(var, space)
            GNVP3SensArgs = [dyn, var, polars2D, "Xfoil",
                             maxiter, timestep,
                             dyn.trim['U'], dyn.trim['AoA']]
            dyn.runAnalysis(runGNVPsensitivityParallel, GNVP3SensArgs)
            dyn.sensResults[var] = pertrResults(
                f"{dyn.CASEDIR}/Sensitivity_{var}", HOMEDIR)
        print("Sensitivity Analysis took : --- %s seconds ---" %
              (time.time() - sens_time))
        dyn.save()
except KeyError:
    print('Plane could not be trimmed')

# print time it took
print("PROGRAM TERMINATED")
print("Execution took : --- %s seconds ---" % (time.time() - start_time))
