import os
from typing import Any

from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

from ICARUS.Core.struct import Struct
from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Database.utils import angle_to_case
from ICARUS.Enviroment.definition import Environment
from ICARUS.Software.GenuVP3.filesInterface import run_gnvp_case
from ICARUS.Software.GenuVP3.postProcess.forces import forces_to_polars
from ICARUS.Software.GenuVP3.utils import define_movements
from ICARUS.Software.GenuVP3.utils import make_surface_dict
from ICARUS.Software.GenuVP3.utils import Movement
from ICARUS.Software.GenuVP3.utils import set_parameters
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import Wing

# from ICARUS.Software.GenuVP3.postProcess.forces import rotateForces


def gnvp_angle_case(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    environment: Environment,
    movements: list[list[Movement]],
    bodies_dicts: list[dict[str, Any]],
    solver_options: dict[str, Any] | Struct,
) -> str:
    """
    Run a single angle simulation in GNVP3

    Args:
        plane (Airplane): Airplane Object
        db (DB): Database Object
        solver2D (str): Name of 2D Solver to be used
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Freestream Velocity Magnitude
        angle (float): Angle of attack in degrees
        environment (Environment): Environment Object
        movements (list[list[Movement]]): List of movements for each surface
        bodies_dicts (list[dict[str, Any]]): Bodies in dict format
        solver_options (dict[str, Any] | Struct): Solver Options

    Returns:
        str: Case Done Message
    """
    HOMEDIR: str = db.HOMEDIR
    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    airfoils: list[str] = plane.airfoils
    foilsDB: Database_2D = db.foilsDB

    print(f"Running Angles {angle}")
    folder: str = angle_to_case(angle)
    CASEDIR: str = os.path.join(PLANEDIR, folder)
    os.makedirs(CASEDIR, exist_ok=True)

    params: dict[str, Any] = set_parameters(
        bodies_dicts,
        plane,
        maxiter,
        timestep,
        u_freestream,
        angle,
        environment,
        solver_options,
    )
    run_gnvp_case(
        CASEDIR,
        HOMEDIR,
        GENUBASE,
        movements,
        bodies_dicts,
        params,
        airfoils,
        foilsDB,
        solver2D,
    )

    return f"Angle {angle} Done"


def run_gnvp_angles(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angles: list[float],
    environment: Environment,
    solver_options: dict[str, Any],
) -> None:
    """Run Multiple Angles Simulation in GNVP3

    Args:
        plane (Airplane): Plane Object
        db (DB): Database
        solver2D (str): Name of 2D Solver to be used for the 2d polars
        maxiter (int): Maxiteration for each case
        timestep (float): Timestep for simulations
        u_freestream (float): Freestream Velocity
        angles (list[float]): List of angles to run
        environment (Environment): Enviroment Object
        solver_options (dict[str, Any]): Solver Options
    """
    movements: list[list[Movement]] = define_movements(
        plane.surfaces,
        plane.CG,
        plane.orientation,
        plane.disturbances,
    )
    bodies_dicts: list[dict[str, Any]] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies_dicts.append(make_surface_dict(surface, i))

    print("Running Angles in Sequential Mode")
    for angle in angles:
        msg: str = gnvp_angle_case(
            plane,
            db,
            solver2D,
            maxiter,
            timestep,
            u_freestream,
            angle,
            environment,
            movements,
            bodies_dicts,
            solver_options,
        )
        print(msg)


def run_gnvp_angles_parallel(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angles: list[float] | ndarray[Any, dtype[floating[Any]]],
    environment: Environment,
    solver_options: dict[str, Any],
) -> None:
    """Run all specified angle simulations in GNVP3 in parallel

    Args:
        plane (Airplane): Plane Object
        db (DB): Database
        solver2D (str): 2D Solver Name to be used for 2d polars
        maxiter (int): Number of max iterations for each simulation
        timestep (float): Timestep between each iteration
        u_freestream (float): Freestream Velocity Magnitude
        angles (list[float] | ndarray[Any, dtype[floating[Any]]]): List of angles to run
        environment (Environment): Environment Object
        solver_options (dict[str, Any]): Solver Options
    """
    movements: list[list[Movement]] = define_movements(
        plane.surfaces,
        plane.CG,
        plane.orientation,
        plane.disturbances,
    )
    bodies: list[dict[str, Any]] = []

    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces
    for i, surface in enumerate(surfaces):
        bodies.append(make_surface_dict(surface, i))

    from multiprocessing import Pool

    print("Running Angles in Parallel Mode")
    with Pool(12) as pool:
        args_list = [
            (
                plane,
                db,
                solver2D,
                maxiter,
                timestep,
                u_freestream,
                angle,
                environment,
                movements,
                bodies,
                solver_options,
            )
            for angle in angles
        ]
        res: list[str] = pool.starmap(gnvp_angle_case, args_list)

        for msg in res:
            print(msg)


def process_gnvp3_angle_run(plane: Airplane, db: DB) -> DataFrame:
    """Procces the results of the GNVP3 AoA Analysis and
    return the forces calculated in a DataFrame

    Args:
        plane (Airplane): Plane Object
        db (DB): Database

    Returns:
        DataFrame: Forces Calculated
    """
    HOMEDIR: str = db.HOMEDIR
    CASEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    forces: DataFrame = forces_to_polars(CASEDIR, HOMEDIR)
    # rotatedforces = rotateForces(forces, forces["AoA"])
    return forces  # , rotatedforces
