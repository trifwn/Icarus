import os

from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.db import DB
from ICARUS.Database.utils import dst2case
from ICARUS.Software.GenuVP3.filesInterface import runGNVPcase
from ICARUS.Software.GenuVP3.postProcess.forces import forces2pertrubRes
from ICARUS.Software.GenuVP3.utils import airMov
from ICARUS.Software.GenuVP3.utils import makeSurfaceDict
from ICARUS.Software.GenuVP3.utils import setParams


def GNVPdstCase(
    plane,
    db,
    solver2D,
    maxiter,
    timestep,
    Uinf,
    angle,
    environment,
    bodies,
    dst,
    analysis,
    solver_options,
):
    HOMEDIR = db.HOMEDIR
    PLANEDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    airfoils = plane.airfoils
    foilsDB = db.foilsDB

    movements = airMov(bodies, plane.CG, plane.orientation, [dst])

    print(f"Running Case {dst.var} - {dst.amplitude}")
    folder = dst2case(dst)
    CASEDIR = os.path.join(PLANEDIR, analysis, folder)
    os.makedirs(CASEDIR, exist_ok=True)

    params = setParams(
        bodies,
        plane,
        maxiter,
        timestep,
        Uinf,
        angle,
        environment,
        solver_options,
    )
    runGNVPcase(
        CASEDIR,
        HOMEDIR,
        GENUBASE,
        movements,
        bodies,
        params,
        airfoils,
        foilsDB.Data,
        solver2D,
    )

    return f"Case {dst.var} : {dst.amplitude} Done"


def runGNVPpertr(
    plane,
    db,
    solver2D,
    maxiter,
    timestep,
    Uinf,
    angles,
    environment,
    solver_options,
):
    bodies = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for dst in plane.disturbances:
        msg = GNVPdstCase(
            plane,
            db,
            solver2D,
            maxiter,
            timestep,
            Uinf,
            angles,
            environment,
            bodies,
            dst,
            "Dynamics",
            solver_options,
        )
        print(msg)


def runGNVPpertrParallel(
    plane,
    environment,
    db,
    solver2D,
    maxiter,
    timestep,
    Uinf,
    angles,
    solver_options,
):
    bodies = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    disturbances = plane.disturbances

    from multiprocessing import Pool

    with Pool(12) as pool:
        args_list = [
            (
                plane,
                db,
                solver2D,
                maxiter,
                timestep,
                Uinf,
                angles,
                environment,
                bodies,
                dst,
                "Dynamics",
                solver_options,
            )
            for dst in disturbances
        ]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def runGNVPsensitivity(
    plane,
    environment,
    var,
    db,
    solver2D,
    maxiter,
    timestep,
    Uinf,
    angles,
    solver_options,
):
    bodies = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for dst in plane.sensitivity[var]:
        msg = GNVPdstCase(
            plane,
            db,
            solver2D,
            maxiter,
            timestep,
            Uinf,
            angles,
            environment,
            bodies,
            dst,
            "Sensitivity",
            solver_options,
        )
        print(msg)


def runGNVPsensitivityParallel(
    plane,
    environment,
    var,
    db,
    solver2D,
    maxiter,
    timestep,
    Uinf,
    angles,
    solver_options,
):
    bodies = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    disturbances = plane.sensitivity[var]

    from multiprocessing import Pool

    with Pool(12) as pool:
        args_list = [
            (
                plane,
                db,
                solver2D,
                maxiter,
                timestep,
                Uinf,
                angles,
                environment,
                bodies,
                dst,
                f"Sensitivity_{dst.var}",
                solver_options,
            )
            for dst in disturbances
        ]

        res = pool.starmap(GNVPdstCase, args_list)
        for msg in res:
            print(msg)


def processGNVPpertrubations(plane, db: DB):
    HOMEDIR = db.HOMEDIR
    DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
    forces = forces2pertrubRes(DYNDIR, HOMEDIR)
    # rotatedforces = rotateForces(forces, forces["AoA"])
    return forces  # rotatedforces


# def processGNVPsensitivity(plane, db: DB):
#     HOMEDIR = db.HOMEDIR
#     DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
#     forces = forces2pertrubRes(DYNDIR, HOMEDIR)
#     # rotatedforces = rotateForces(forces, forces["AoA"])
#     return forces #rotatedforces
