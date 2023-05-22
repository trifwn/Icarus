import os
from typing import Any

from pandas import DataFrame

from ICARUS.Core.struct import Struct
from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Database.utils import disturbance_to_case
from ICARUS.Enviroment.definition import Environment
from ICARUS.Flight_Dynamics.disturbances import Disturbance
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Software.GenuVP3.filesInterface import run_gnvp_case
from ICARUS.Software.GenuVP3.postProcess.forces import forces_to_pertrubation_results
from ICARUS.Software.GenuVP3.utils import define_movements
from ICARUS.Software.GenuVP3.utils import make_surface_dict
from ICARUS.Software.GenuVP3.utils import Movement
from ICARUS.Software.GenuVP3.utils import set_parameters
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing import Wing


def gnvp_disturbance_case(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    environment: Environment,
    surfaces: list[Wing],
    bodies_dict: list[dict[str, Any]],
    dst: Disturbance,
    analysis: str,
    solver_options: dict[str, Any] | Struct,
) -> str:
    """
    Run a single disturbance simulation in GNVP3

    Args:
        plane (Airplane): Plane Object
        db (DB): Database Object
        solver2D (str): Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Freestream Velocity magnitude
        angle (float): Angle of attack in degrees
        environment (Environment): Environment Object
        surfaces (list[Wing]): List of surfaces
        bodies_dict (list[dict[str, Any]]): List of bodies in dict format
        dst (Disturbance): Disturbance to be run
        analysis (str): Analysis Name
        solver_options (dict[str, Any] | Struct): Solver Options

    Returns:
        str: Case Done Message
    """
    HOMEDIR: str = db.HOMEDIR
    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    airfoils: list[str] = plane.airfoils
    foilsDB: Database_2D = db.foilsDB

    movements: list[list[Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
        [dst],
    )

    print(f"Running Case {dst.var} - {dst.amplitude}")
    folder: str = disturbance_to_case(dst)
    CASEDIR: str = os.path.join(PLANEDIR, analysis, folder)
    os.makedirs(CASEDIR, exist_ok=True)

    params: dict[str, Any] = set_parameters(
        bodies_dict,
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
        bodies_dict,
        params,
        airfoils,
        foilsDB,
        solver2D,
    )

    return f"Case {dst.var} : {dst.amplitude} Done"


def run_pertrubation_serial(
    plane: Airplane,
    state: State,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle_of_attack: float,
    environment: Environment,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is serial.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        db (DB): Database Object
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float):  Freestream Velocity magnitude
        angle_of_attack (float): Angle of attack in degrees
        environment (Environment): Environment Object
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[dict[str, Any]] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies_dicts.append(make_surface_dict(surface, i))

    for dst in state.disturbances:
        msg: str = gnvp_disturbance_case(
            plane,
            db,
            solver2D,
            maxiter,
            timestep,
            u_freestream,
            angle_of_attack,
            environment,
            surfaces,
            bodies_dicts,
            dst,
            "Dynamics",
            solver_options,
        )
        print(msg)


def run_pertrubation_parallel(
    plane: Airplane,
    state: State,
    environment: Environment,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is parallel.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        environment (Environment): Environment Object
        db (DB): Database Object
        solver2D (str): Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Freestream Velocity magnitude
        angle_of_attack (float): Angle of attack in degrees
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[dict[str, Any]] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies_dicts.append(make_surface_dict(surface, i))

    disturbances: list[Disturbance] = state.disturbances

    from multiprocessing import Pool

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
                surfaces,
                bodies_dicts,
                dst,
                "Dynamics",
                solver_options,
            )
            for dst in disturbances
        ]

        res: list[str] = pool.starmap(gnvp_disturbance_case, args_list)
        for msg in res:
            print(msg)


def sensitivity_serial(
    plane: Airplane,
    state: State,
    environment: Environment,
    var: str,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is serial.
    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        environment (Environment): Environment Object
        var (str): Variable to be perturbed
        db (DB): Database Object
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Velocity Magnitude
        angle_of_attack (float): Angle of attack in degrees
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[dict[str, Any]] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies_dicts.append(make_surface_dict(surface, i))

    for dst in state.sensitivity[var]:
        msg: str = gnvp_disturbance_case(
            plane,
            db,
            solver2D,
            maxiter,
            timestep,
            u_freestream,
            angle,
            environment,
            surfaces,
            bodies_dicts,
            dst,
            "Sensitivity",
            solver_options,
        )
        print(msg)


def sensitivity_parallel(
    plane: Airplane,
    state: State,
    environment: Environment,
    var: str,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is parallel.
    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        environment (Environment): Environment Object
        var (str): Variable to be perturbed
        db (DB): Database Object
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Velocity Magnitude
        angle_of_attack (float): Angle of attack in degrees
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[dict[str, Any]] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        bodies_dicts.append(make_surface_dict(surface, i))

    disturbances: list[Disturbance] = state.sensitivity[var]

    from multiprocessing import Pool

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
                surfaces,
                bodies_dicts,
                dst,
                f"Sensitivity_{dst.var}",
                solver_options,
            )
            for dst in disturbances
        ]

        res: list[str] = pool.starmap(gnvp_disturbance_case, args_list)
        for msg in res:
            print(msg)


def proccess_pertrubation_res(plane: Airplane, db: DB) -> DataFrame:
    """
    Process the pertrubation results from the GNVP3 solver

    Args:
        plane (Airplane | Dynamic_Airplane): Airplane Object
        db (DB): Database Object

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation
    """
    HOMEDIR: str = db.HOMEDIR
    DYNDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
    forces: DataFrame = forces_to_pertrubation_results(DYNDIR, HOMEDIR)
    # rotatedforces = rotateForces(forces, forces["AoA"])
    return forces  # rotatedforces


# def processGNVPsensitivity(plane, db: DB):
#     HOMEDIR = db.HOMEDIR
#     DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
#     forces = forces2pertrubRes(DYNDIR, HOMEDIR)
#     # rotatedforces = rotateForces(forces, forces["AoA"])
#     return forces #rotatedforces
