import os

from ICARUS.Software.GenuVP3.filesInterface import runGNVPcase
from ICARUS.Software.GenuVP3.postProcess.forces import forces2pertrubRes
from ICARUS.Software.GenuVP3.utils import setParams, airMov, makeSurfaceDict

from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.utils import dst2case
from ICARUS.Database.db import DB

def GNVPdstCase(plane,environment, db, solver2D, maxiter, timestep, Uinf, angle, bodies, dst, analysis):
    HOMEDIR = db.HOMEDIR
    PLANEDIR = os.path.join(db.vehiclesDB.DATADIR ,plane.CASEDIR)
    airfoils = plane.airfoils
    dens = environment.AirDensity
    foilsDB = db.foilsDB

    movements = airMov(plane.surfaces, plane.CG,
                       plane.orientation, [dst])

    print(f"Running Case {dst.var} - {dst.amplitude}")

    # if make distubance folder
    folder = dst2case(dst)
    CASEDIR = os.path.join(PLANEDIR, analysis, folder)
    os.makedirs(CASEDIR,exist_ok=True)

    params = setParams(len(bodies), len(airfoils), maxiter, timestep,
                       Uinf, angle, dens)
    runGNVPcase(CASEDIR, HOMEDIR, GENUBASE, movements,
                bodies, params, airfoils, foilsDB.Data, solver2D)

    return f"Case {dst.var} : {dst.amplitude} Done"


def runGNVPpertr(plane,environment, db, solver2D, maxiter, timestep, Uinf, angles):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.disturbances:
        msg = GNVPdstCase(plane,environment, db, solver2D, maxiter, timestep,
                          Uinf, angles, bodies, dst, "Dynamics")
        print(msg)


def runGNVPpertrParallel(plane,environment, db, solver2D, maxiter, timestep, Uinf, angles):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))
    disturbances = plane.disturbances
    with Pool(12) as pool:
        args_list = [(plane,environment, db, solver2D, maxiter, timestep,
                      Uinf, angles, bodies, dst, "Dynamics") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def runGNVPsensitivity(plane,environment, var, db, solver2D, maxiter, timestep, Uinf, angles):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.sensitivity[var]:
        msg = GNVPdstCase(plane,environment, db, solver2D, maxiter, timestep,
                          Uinf, angles, bodies, dst, "Sensitivity")
        print(msg)


def runGNVPsensitivityParallel(plane,environment, var, db, solver2D, maxiter, timestep, Uinf, angles):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    disturbances = plane.sensitivity[var]
    with Pool(12) as pool:
        args_list = [(plane,environment, db, solver2D, maxiter, timestep,
                      Uinf, angles, bodies, dst, f"Sensitivity_{dst.var}") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)
            
def processGNVPpertrubations(plane, db: DB):
    HOMEDIR = db.HOMEDIR
    DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
    forces = forces2pertrubRes(DYNDIR, HOMEDIR)
    # rotatedforces = rotateForces(forces, forces["AoA"])
    return forces #rotatedforces

# def processGNVPsensitivity(plane, db: DB):
#     HOMEDIR = db.HOMEDIR
#     DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
#     forces = forces2pertrubRes(DYNDIR, HOMEDIR)
#     # rotatedforces = rotateForces(forces, forces["AoA"])
#     return forces #rotatedforces