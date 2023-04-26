import os

from .filesInterface import runGNVPcase
from .utils import setParams, airMov, makeSurfaceDict
from ICARUS.Database import BASEGNVP3 as GENUBASE

from ICARUS.Database.Database_3D import dst2case


def GNVPdstCase(plane, db, solver2D, maxiter, timestep, Uinf, angle, bodies, dst, analysis):
    HOMEDIR = db.HOMEDIR
    PLANEDIR = os.path.join(db.vehiclesDB.DATADIR ,plane.CASEDIR)
    airfoils = plane.airfoils
    dens = plane.dens
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


def runGNVPpertr(plane, db, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.disturbances:
        msg = GNVPdstCase(plane, db, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Dynamics")
        print(msg)


def runGNVPpertrParallel(plane, db, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))
    disturbances = plane.disturbances
    with Pool(12) as pool:
        args_list = [(plane, db, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, "Dynamics") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def runGNVPsensitivity(plane, var, db, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.sensitivity[var]:
        msg = GNVPdstCase(plane.pln, db, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Sensitivity")
        print(msg)


def runGNVPsensitivityParallel(plane, var, db, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    disturbances = plane.sensitivity[var]
    with Pool(12) as pool:
        args_list = [(plane.pln, db, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, f"Sensitivity_{dst.var}") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)
