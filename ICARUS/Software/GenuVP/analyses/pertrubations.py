import os
import re
from threading import Thread
from typing import Any

from pandas import DataFrame
from tqdm.auto import tqdm

from ICARUS.Core.struct import Struct
from ICARUS.Database import BASEGNVP3 as GENUBASE
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Database.utils import disturbance_to_case
from ICARUS.Enviroment.definition import Environment
from ICARUS.Flight_Dynamics.disturbances import Disturbance
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Software.GenuVP.analyses.monitor_progress import parallel_monitor
from ICARUS.Software.GenuVP.analyses.monitor_progress import serial_monitor
from ICARUS.Software.GenuVP.files.gnvp3_interface import run_gnvp3_case
from ICARUS.Software.GenuVP.post_process import progress
from ICARUS.Software.GenuVP.post_process.forces import forces_to_pertrubation_results
from ICARUS.Software.GenuVP.post_process.forces import rotate_forces
from ICARUS.Software.GenuVP.utils import define_movements
from ICARUS.Software.GenuVP.utils import make_surface_dict
from ICARUS.Software.GenuVP.utils import Movement
from ICARUS.Software.GenuVP.utils import set_parameters
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
    run_gnvp3_case(
        CASEDIR,
        HOMEDIR,
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

    progress_bars = []
    for i, dst in enumerate(state.disturbances):
        job = Thread(
            target=gnvp_disturbance_case,
            kwargs={
                "plane": plane,
                "db": db,
                "solver2D": solver2D,
                "maxiter": maxiter,
                "timestep": timestep,
                "u_freestream": u_freestream,
                "angle": angle_of_attack,
                "environment": environment,
                "surfaces": surfaces,
                "bodies_dicts": bodies_dicts,
                "dst": dst,
                "analysis": "Dynamics",
                "solver_options": solver_options,
            },
        )
        pbar = tqdm(
            total=1,
            desc=f"DST:{dst.var} - {dst.amplitude}",
            position=i,
            leave=True,
            colour=" #008080",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)
        folder: str = disturbance_to_case(dst)
        PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
        CASEDIR: str = os.path.join(PLANEDIR, "Dynamics", folder)
        job_monitor = Thread(
            target=serial_monitor,
            kwargs={
                "progress_bars": progress_bars,
                "CASEDIR": CASEDIR,
                "position": i,
                "lock": None,
                "max_iter": maxiter,
                "refresh_progress": 2,
            },
        )

        # Start the job
        job.start()
        job_monitor.start()

        # Wait for the job to finish
        job.join()
        job_monitor.join()


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

    def run() -> None:
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

            _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)

    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    folders: list[str] = [disturbance_to_case(dst) for dst in disturbances]
    CASEDIRS: list[str] = [os.path.join(PLANEDIR, "Dynamics", folder) for folder in folders]

    refresh_progress: float = 2
    job = Thread(target=run)
    job_monitor = Thread(
        target=parallel_monitor,
        kwargs={
            "CASEDIRS": CASEDIRS,
            "variables": [f"{dst.var} - {dst.amplitude}" for dst in disturbances],
            "max_iter": maxiter,
            "refresh_progress": refresh_progress,
        },
    )

    # Start the job
    job.start()
    job_monitor.start()

    # Wait for the job to finish
    job.join()
    job_monitor.join()


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

        _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)


def proccess_pertrubation_res(plane: Airplane, db: DB, state: State) -> DataFrame:
    """
    Process the pertrubation results from the GNVP solver

    Args:
        plane (Airplane | Dynamic_Airplane): Airplane Object
        db (DB): Database Object
        state (State): Plane State to load results to

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation
    """
    HOMEDIR: str = db.HOMEDIR
    DYNDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
    forces: DataFrame = forces_to_pertrubation_results(DYNDIR, HOMEDIR)

    state.set_pertrubation_results(forces)
    state.stability_fd(polar_name='2D')

    return forces


# def processGNVPsensitivity(plane, db: DB):
#     HOMEDIR = db.HOMEDIR
#     DYNDIR = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
#     forces = forces2pertrubRes(DYNDIR, HOMEDIR)
#     # rotatedforces = rotateForces(forces, forces["AoA"])
#     return forces #rotatedforces
