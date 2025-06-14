from __future__ import annotations

import logging
import os
from subprocess import CalledProcessError
from threading import Event
from threading import Thread
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NoReturn

from pandas import DataFrame
from tqdm import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.database import angle_to_directory

from ..files import gnvp3_case
from ..files import gnvp7_case
from ..post_process import log_forces
from ..utils import GenuParameters
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from ..utils import define_movements
from .monitor_progress import parallel_monitor
from .monitor_progress import serial_monitor

if TYPE_CHECKING:
    from ICARUS.environment import Environment
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface


class StopRunningThreadError(Exception):
    pass


def gnvp_polars(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angles: list[float] | FloatArray,
    solver_options: dict[str, Any] | Struct,
    parallel: bool = False,
    gnvp_version: Literal[3, 7] = 3,
) -> None:
    """Run Polar Simulation in GNVP3 or GNVP7

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        solver2D (str): Name of 2D Solver to be used for the 2d polars
        maxiter (int): Maxiteration for each case
        timestep (float): Timestep for simulations
        angles (list[float]): List of angles to run
        solver_options (dict[str, Any]): Solver Options
        parallel (bool, optional): Run in parallel. Defaults to False.
        gnvp_version (int, optional): Version of GenuVP solver. Defaults to 3.

    """
    if parallel:
        gnvp_polars_parallel(
            plane=plane,
            state=state,
            solver2D=solver2D,
            maxiter=maxiter,
            timestep=timestep,
            angles=angles,
            gnvp_version=gnvp_version,
            solver_options=solver_options,
        )
    else:
        gnvp_polars_serial(
            plane=plane,
            state=state,
            solver2D=solver2D,
            maxiter=maxiter,
            timestep=timestep,
            angles=angles,
            gnvp_version=gnvp_version,
            solver_options=solver_options,
        )


def gnvp3_polars(*args: Any, **kwargs: Any) -> None:
    gnvp_polars_serial(gnvp_version=3, *args, **kwargs)  # type: ignore


def gnvp7_polars(*args: Any, **kwargs: Any) -> None:
    gnvp_polars_serial(gnvp_version=7, *args, **kwargs)  # type: ignore


def gnvp3_polars_parallel(*args: Any, **kwargs: Any) -> None:
    gnvp_polars_parallel(gnvp_version=3, *args, **kwargs)  # type: ignore


def gnvp7_polars_parallel(*args: Any, **kwargs: Any) -> None:
    gnvp_polars_parallel(gnvp_version=7, *args, **kwargs)  # type: ignore


def gnvp_aoa_case(
    DB: Database,
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    environment: Environment,
    movements: list[list[GNVP_Movement]],
    bodies_dicts: list[GenuSurface],
    gnvp_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """Run a single angle simulation in GNVP3

    Args:
        plane (Airplane): Airplane Object
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
    PLANEDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )

    case_directory: str = os.path.join(PLANEDIR, angle_to_directory(angle))
    os.makedirs(case_directory, exist_ok=True)

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
    if gnvp_version == 7:
        run = gnvp7_case
    else:
        run = gnvp3_case

    run(
        case_directory=case_directory,
        movements=movements,
        genu_bodies=bodies_dicts,
        params=params,
        solver2D=solver2D,
    )


def gnvp_polars_serial(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angles: list[float] | FloatArray,
    gnvp_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """Run Multiple Angles Simulation in GNVP3

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        solver2D (str): Name of 2D Solver to be used for the 2d polars
        maxiter (int): Maxiteration for each case
        timestep (float): Timestep for simulations
        angles (list[float]): List of angles to run
        gnvp_version (int): Version of GenuVP solver
        solver_options (dict[str, Any]): Solver Options

    """
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        gen_surf: GenuSurface = GenuSurface(surface, i)
        bodies_dicts.append(gen_surf)

    movements: list[list[GNVP_Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
    )
    print("Running Polars Serially")
    DB = Database.get_instance()
    PLANEDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )
    progress_bars: list[tqdm[NoReturn]] = []
    for i, angle in enumerate(angles):
        folder: str = angle_to_directory(angle)
        CASEDIR: str = os.path.join(PLANEDIR, folder)

        job = Thread(
            target=gnvp_aoa_case,
            kwargs={
                "DB": DB,
                "plane": plane,
                "state": state,
                "solver2D": solver2D,
                "maxiter": maxiter,
                "timestep": timestep,
                "u_freestream": state.u_freestream,
                "angle": angle,
                "environment": state.environment,
                "movements": movements,
                "bodies_dicts": bodies_dicts,
                "gnvp_version": gnvp_version,
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
                "gnvp_version": gnvp_version,
            },
        )

        # Start
        job.start()
        job_monitor.start()

        # Join
        job.join()
        job_monitor.join()
    GenuSurface.airfoil_names = {}
    GenuSurface.surf_names = {}


def gnvp_polars_parallel(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angles: list[float] | FloatArray,
    gnvp_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """Run all specified angle simulations in GNVP3 in parallel

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        solver2D (str): 2D Solver Name to be used for 2d polars
        maxiter (int): Number of max iterations for each simulation
        timestep (float): Timestep between each iteration
        angles (list[float] | FloatArray): List of angles to run
        solver_options (dict[str, Any]): Solver Options

    """
    DB = Database.get_instance()

    bodies_dict: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces
    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dict.append(genu_surf)

    movements: list[list[GNVP_Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
    )

    stop_event = Event()
    from multiprocessing import Pool

    print("Running Polars in Parallel")

    def run() -> None:
        if gnvp_version == 3:
            num_processes = int(CPU_TO_USE)
        else:
            num_processes = int(CPU_TO_USE)
        with Pool(num_processes) as pool:
            args_list = [
                (
                    DB,
                    plane,
                    state,
                    solver2D,
                    maxiter,
                    timestep,
                    state.u_freestream,
                    angle,
                    state.environment,
                    movements,
                    bodies_dict,
                    gnvp_version,
                    solver_options,
                )
                for angle in angles
            ]

            try:
                _ = pool.starmap(gnvp_aoa_case, args_list)

            except CalledProcessError as e:
                print(f"Could not run GNVP got: {e}")
                stop_event.set()

    PLANEDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )
    folders: list[str] = [angle_to_directory(angle) for angle in angles]
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
            "gnvp_version": gnvp_version,
            "stop_event": stop_event,
        },
    )

    # Start the threads and catch stopRunningThreadError to stop each one if it fails
    job.start()
    job_monitor.start()

    job.join()
    job_monitor.join()


def process_gnvp_polars_3(plane: Airplane, state: State) -> DataFrame:
    return process_gnvp_polars(plane, state, 3)


def process_gnvp_polars_7(plane: Airplane, state: State) -> DataFrame:
    return process_gnvp_polars(plane, state, 7)


def process_gnvp_polars(
    plane: Airplane,
    state: State,
    gnvp_version: int,
) -> DataFrame:
    """Procces the results of the GNVP3 AoA Analysis and
    return the forces calculated in a DataFrame

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        gnvp_version: GNVP Version

    Returns:
        DataFrame: Forces Calculated

    """
    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )

    forces: DataFrame = log_forces(CASEDIR, gnvp_version)
    plane.save()
    state.add_polar(
        polar=forces,
        polar_prefix=f"GenuVP{gnvp_version}",
        is_dimensional=True,
    )
    state.save(CASEDIR)

    logging.info("Adding Results to Database")
    # Add Plane to Database
    file_plane: str = os.path.join(CASEDIR, f"{plane.name}.json")
    _ = DB.load_vehicle(name=plane.name, file=file_plane)

    # Add Results to Database
    DB.load_vehicle_solver_data(
        vehicle=plane,
        state=state,
        folder=CASEDIR,
        solver=f"GenuVP{gnvp_version}",
    )

    return forces
