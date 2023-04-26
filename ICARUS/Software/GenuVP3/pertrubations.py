import os

from .filesInterface import runGNVPcase
from .utils import setParams, airMov, makeSurfaceDict
from ICARUS.Database import BASEGNVP3 as GENUBASE

from ICARUS.Database.Database_3D import dst2case


def GNVPdstCase(plane, foildb, solver2D, maxiter, timestep, Uinf, angle, bodies, dst, analysis):
    PLANEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    dens = plane.dens

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
                bodies, params, airfoils, foildb.Data, solver2D)

    return f"Case {dst.var} : {dst.amplitude} Done"


def runGNVPpertr(plane, foildb, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.disturbances:
        msg = GNVPdstCase(plane, foildb.Data, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Dynamics")
        print(msg)


def runGNVPpertrParallel(plane, foildb, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))
    disturbances = plane.disturbances
    with Pool(12) as pool:
        args_list = [(plane, foildb.Data, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, "Dynamics") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def runGNVPsensitivity(plane, var, foildb, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.sensitivity[var]:
        msg = GNVPdstCase(plane.pln, foildb.Data, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Sensitivity")
        print(msg)


def runGNVPsensitivityParallel(plane, var, foildb, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    disturbances = plane.sensitivity[var]
    with Pool(12) as pool:
        args_list = [(plane.pln, foildb.Data, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, f"Sensitivity_{dst.var}") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)
