import os
from threading import Thread
from typing import Any

from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from ICARUS.Core.struct import Struct
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Database.utils import angle_to_case
from ICARUS.Enviroment.definition import Environment
from ICARUS.Input_Output.GenuVP.analyses.monitor_progress import parallel_monitor
from ICARUS.Input_Output.GenuVP.analyses.monitor_progress import serial_monitor
from ICARUS.Input_Output.GenuVP.files.gnvp3_interface import run_gnvp3_case
from ICARUS.Input_Output.GenuVP.files.gnvp7_interface import run_gnvp7_case
from ICARUS.Input_Output.GenuVP.post_process.forces import log_forces
from ICARUS.Input_Output.GenuVP.utils.genu_movement import define_movements
from ICARUS.Input_Output.GenuVP.utils.genu_movement import Movement
from ICARUS.Input_Output.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.Input_Output.GenuVP.utils.genu_surface import GenuSurface
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.wing_segment import Wing_Segment

# from ICARUS.Software.GenuVP.utils.utils import make_surface_dict


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
    bodies_dicts: list[GenuSurface],
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
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
        bodies_dicts (list[GenuSurface]): Bodies in Genu Format
        solver_options (dict[str, Any] | Struct): Solver Options

    Returns:
        str: Case Done Message
    """
    HOMEDIR: str = db.HOMEDIR
    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    airfoils: list[str] = plane.airfoils
    foilsDB: Database_2D = db.foilsDB

    folder: str = angle_to_case(angle)
    CASEDIR: str = os.path.join(PLANEDIR, folder)
    os.makedirs(CASEDIR, exist_ok=True)

    params: GenuParameters = GenuParameters(
        bodies_dicts,
        plane,
        maxiter,
        timestep,
        u_freestream,
        angle,
        environment,
        solver_options,
    )
    if genu_version == 7:
        run = run_gnvp7_case
    else:
        run = run_gnvp3_case

    run(
        CASEDIR=CASEDIR,
        HOMEDIR=HOMEDIR,
        movements=movements,
        bodies_dicts=bodies_dicts,
        params=params,
        airfoils=airfoils,
        foildb=foilsDB,
        solver2D=solver2D,
    )


def run_gnvp3_angles(*args: Any, **kwargs: Any) -> None:
    run_gnvp_angles(genu_version=3, *args, **kwargs)  # type: ignore


def run_gnvp7_angles(*args: Any, **kwargs: Any) -> None:
    run_gnvp_angles(genu_version=7, *args, **kwargs)  # type: ignore


def run_gnvp3_angles_parallel(*args: Any, **kwargs: Any) -> None:
    run_gnvp_angles_parallel(genu_version=3, *args, **kwargs)  # type: ignore


def run_gnvp7_angles_parallel(*args: Any, **kwargs: Any) -> None:
    run_gnvp_angles_parallel(genu_version=7, *args, **kwargs)  # type: ignore


def run_gnvp_angles(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angles: list[float],
    environment: Environment,
    genu_version: int,
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
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing_Segment] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        gen_surf: GenuSurface = GenuSurface(surface, i)
        bodies_dicts.append(gen_surf)

    movements: list[list[Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
        plane.disturbances,
    )
    print("Running Angles in Sequential Mode")

    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    progress_bars: list[tqdm] = []
    for i, angle in enumerate(angles):
        folder: str = angle_to_case(angle)
        CASEDIR: str = os.path.join(PLANEDIR, folder)

        job = Thread(
            target=gnvp_angle_case,
            kwargs={
                "plane": plane,
                "db": db,
                "solver2D": solver2D,
                "maxiter": maxiter,
                "timestep": timestep,
                "u_freestream": u_freestream,
                "angle": angle,
                "environment": environment,
                "movements": movements,
                "bodies_dicts": bodies_dicts,
                "genu_version": genu_version,
                "solver_options": solver_options,
            },
        )
        pbar = tqdm(
            total=maxiter,
            desc=f"{angle}:",
            position=i,
            leave=True,
            colour="#cc3300",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)

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

        # Start
        job.start()
        job_monitor.start()

        # Join
        job.join()
        job_monitor.join()


def run_gnvp_angles_parallel(
    plane: Airplane,
    db: DB,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angles: list[float] | ndarray[Any, dtype[floating[Any]]],
    environment: Environment,
    genu_version: int,
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
    bodies_dict: list[GenuSurface] = []

    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[Wing_Segment] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces
    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dict.append(genu_surf)

    movements: list[list[Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
        plane.disturbances,
    )
    from multiprocessing import Pool

    print("Running Angles in Parallel Mode")

    def run() -> None:
        if genu_version == 3:
            num_processes = 12
        else:
            num_processes = 3
        with Pool(num_processes) as pool:
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
                    bodies_dict,
                    genu_version,
                    solver_options,
                )
                for angle in angles
            ]
            pool.starmap(gnvp_angle_case, args_list)

    PLANEDIR: str = os.path.join(db.vehiclesDB.DATADIR, plane.CASEDIR)
    folders: list[str] = [angle_to_case(angle) for angle in angles]
    CASEDIRS: list[str] = [os.path.join(PLANEDIR, folder) for folder in folders]

    refresh_pogress: float = 2

    job = Thread(target=run)
    job_monitor = Thread(
        target=parallel_monitor,
        kwargs={
            "CASEDIRS": CASEDIRS,
            "variables": angles,
            "max_iter": maxiter,
            "refresh_progress": refresh_pogress,
        },
    )

    # Start
    job.start()
    job_monitor.start()

    # Join
    job.join()
    job_monitor.join()


def process_gnvp_angles_run(plane: Airplane, db: DB) -> DataFrame:
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
    forces: DataFrame = log_forces(CASEDIR, HOMEDIR)
    # rotatedforces: DataFrame = rotate_forces(forces, forces["AoA"])
    return forces