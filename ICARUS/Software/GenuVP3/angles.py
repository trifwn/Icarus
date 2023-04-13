import os

from .interface import runGNVPcase
from .utils import setParams, airMov, makeSurfaceDict
from ICARUS.Database import BASEGNVP3 as GENUBASE

from ICARUS.Database.Database_3D import ang2case


def GNVPangleCase(plane, polars, solver2D, maxiter, timestep, Uinf, angle, dens, movements, bodies):
    HOMEDIR = plane.HOMEDIR
    PLANEDIR = plane.CASEDIR
    airfoils = plane.airfoils
    print(f"Running Angles {angle}")
    folder = ang2case(angle)
    CASEDIR = f"{PLANEDIR}/{folder}"
    os.system(f"mkdir -p {CASEDIR}")
    params = setParams(len(bodies), len(airfoils), maxiter,
                       timestep, Uinf, angle, dens)
    runGNVPcase(CASEDIR, HOMEDIR, GENUBASE, movements,
                bodies, params, airfoils, polars, solver2D)

    return f"Angle {angle} Done"


def runGNVPangles(plane, polars, solver2D, maxiter, timestep, Uinf, angles, dens=1.225):
    plane.defineSim(Uinf, dens)

    movements = airMov(plane.surfaces, plane.CG,
                       plane.orientation, plane.disturbances)
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    print("Running Angles in Sequential Mode")
    for angle in angles:
        msg = GNVPangleCase(plane, polars, solver2D, maxiter, timestep,
                            Uinf, angle, dens, movements, bodies)
        print(msg)


def runGNVPanglesParallel(plane, polars, solver2D, maxiter, timestep, Uinf, angles, dens=1.225):
    from multiprocessing import Pool

    plane.defineSim(Uinf, dens)

    movements = airMov(plane.surfaces, plane.CG,
                       plane.orientation, plane.disturbances)
    bodies = []
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i, plane.CG))

    print("Running Angles in Parallel Mode")
    with Pool(12) as pool:
        args_list = [(plane, polars, solver2D, maxiter, timestep,
                      Uinf, angle, dens, movements, bodies) for angle in angles]
        res = pool.starmap(GNVPangleCase, args_list)

        for msg in res:
            print(msg)
