import os

from .filesInterface import runGNVPcase
from .utils import setParams, airMov, makeSurfaceDict
from ICARUS.Database import BASEGNVP3 as GENUBASE

from ICARUS.Database.Database_3D import dst2case


def GNVPdstCase(plane, polars, solver2D, maxiter, timestep, Uinf, angle, bodies, dst, analysis):
    PLANEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    dens = plane.dens

    movements = airMov(plane.surfaces, plane.CG,
                       plane.orientation, [dst])

    print(f"Running Case {dst.var} - {dst.amplitude}")

    # if make distubance folder
    folder = dst2case(dst)
    CASEDIR = f"{PLANEDIR}/{analysis}/{folder}/"
    os.system(f"mkdir -p {CASEDIR}")

    params = setParams(len(bodies), len(airfoils), maxiter, timestep,
                       Uinf, angle, dens)
    runGNVPcase(CASEDIR, HOMEDIR, GENUBASE, movements,
                bodies, params, airfoils, polars, solver2D)

    return f"Case {dst.var} : {dst.amplitude} Done"


def runGNVPpertr(plane, polars, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.disturbances:
        msg = GNVPdstCase(plane, polars, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Dynamics")
        print(msg)


def runGNVPpertrParallel(plane, polars, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))
    disturbances = plane.disturbances
    with Pool(12) as pool:
        args_list = [(plane, polars, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, "Dynamics") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def runGNVPsensitivity(plane, var, polars, solver2D, maxiter, timestep, Uinf, angle):
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    for dst in plane.sensitivity[var]:
        msg = GNVPdstCase(plane.pln, polars, solver2D, maxiter, timestep,
                          Uinf, angle, bodies, dst, "Sensitivity")
        print(msg)


def runGNVPsensitivityParallel(plane, var, polars, solver2D, maxiter, timestep, Uinf, angle):
    from multiprocessing import Pool

    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    disturbances = plane.sensitivity[var]
    with Pool(12) as pool:
        args_list = [(plane.pln, polars, solver2D, maxiter, timestep,
                      Uinf, angle, bodies, dst, f"Sensitivity_{dst.var}") for dst in disturbances]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)
