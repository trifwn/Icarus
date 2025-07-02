from __future__ import annotations

import os
from threading import Thread
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from pandas import DataFrame
from tqdm.auto import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.database import Database
from ICARUS.database import disturbance_to_directory

from ..files import gnvp3_case
from ..files import gnvp7_case
from ..post_process import forces_to_pertrubation_results
from ..utils import GenuParameters
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from ..utils import define_movements
from .monitor_progress import parallel_monitor
from .monitor_progress import serial_monitor

if TYPE_CHECKING:
    from ICARUS.environment import Environment
    from ICARUS.flight_dynamics import Disturbance
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface


def gnvp_disturbance_case(
    DB: Database,
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    environment: Environment,
    surfaces: Sequence[WingSurface],
    bodies_dicts: list[GenuSurface],
    dst: Disturbance,
    analysis: str,
    gnvp_version: int,
    solver_parameters: dict[str, Any],
) -> str:
    """Run a single disturbance simulation in GNVP3

    Args:
        DB (Database): Database Object
        plane (Airplane): Plane Object
        state (State): Plane State Object
        solver2D (str): Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        u_freestream (float): Freestream Velocity magnitude
        angle (float): Angle of attack in degrees
        environment (Environment): Environment Object
        surfaces (list[Wing]): List of surfaces
        bodies_dicts (list[GenuSurface]): List of bodies in GenuSurface format
        dst (Disturbance): Disturbance to be run
        analysis (str): Analysis Name
        gnvp_version (int): GenuVP version
        solver_parameters (dict[str, Any] | Struct): Solver Options

    Returns:
        str: Case Done Message

    """
    DB = Database.get_instance()
    PLANEDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )

    movements: list[list[GNVP_Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
        [dst],
    )

    folder: str = disturbance_to_directory(dst)
    CASEDIR: str = os.path.join(PLANEDIR, analysis, folder)
    os.makedirs(CASEDIR, exist_ok=True)

    params: GenuParameters = GenuParameters(
        bodies_dicts,
        plane,
        maxiter,
        timestep,
        u_freestream,
        angle,
        environment,
        solver_parameters,
    )

    if gnvp_version == 7:
        run = gnvp7_case
    else:
        run = gnvp3_case

    run(
        CASEDIR,
        movements,
        bodies_dicts,
        params,
        solver2D,
    )

    return f"Case {dst.var} : {dst.amplitude} Done"


def gnvp3_dynamics_serial(*args: Any, **kwars: Any) -> None:
    gnvp_dynamics_serial(gnvp_version=3, *args, **kwars)  # type: ignore


def gnvp7_dynamics_serial(*args: Any, **kwars: Any) -> None:
    gnvp_dynamics_serial(gnvp_version=7, *args, **kwars)  # type: ignore


def gnvp3_dynamics_parallel(*args: Any, **kwars: Any) -> None:
    gnvp_dynamics_parallel(gnvp_version=3, *args, **kwars)  # type: ignore


def gnvp7_dynamics_parallel(*args: Any, **kwars: Any) -> None:
    gnvp_dynamics_parallel(gnvp_version=7, *args, **kwars)  # type: ignore


def gnvp_dynamics_serial(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    gnvp_version: int,
    solver_parameters: dict[str, Any],
) -> None:
    """For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is serial.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        solver_parameters (dict[str, Any]): Solver Options

    """
    DB = Database.get_instance()
    bodies_dicts: list[GenuSurface] = []
    if solver_parameters["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    progress_bars = []
    for i, dst in enumerate(state.disturbances):
        job = Thread(
            target=gnvp_disturbance_case,
            kwargs={
                "DB": DB,
                "plane": plane,
                "state": state,
                "solver2D": solver2D,
                "maxiter": maxiter,
                "timestep": timestep,
                "u_freestream": state.trim["U"],
                "angle": state.trim["AoA"],
                "environment": state.environment,
                "surfaces": surfaces,
                "bodies_dicts": bodies_dicts,
                "dst": dst,
                "analysis": "Dynamics",
                "gnvp_version": gnvp_version,
                "solver_parameters": solver_parameters,
            },
        )
        pbar = tqdm(
            total=maxiter,
            desc=f"DST:{dst.var} - {dst.amplitude:.4f}" if dst.amplitude else f"DST:{dst.var}",
            position=i,
            leave=True,
            colour="RED",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)
        folder: str = disturbance_to_directory(dst)
        PLANEDIR: str = DB.get_vehicle_case_directory(
            airplane=plane,
            state=state,
            solver=f"GenuVP{gnvp_version}",
        )
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
                "gnvp_version": gnvp_version,
            },
        )

        # Start the job
        job.start()
        job_monitor.start()

        # Wait for the job to finish
        job.join()
        job_monitor.join()


def gnvp_dynamics_parallel(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    gnvp_version: int,
    solver_parameters: dict[str, Any],
) -> None:
    """For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is parallel.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        solver2D (str): Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        solver_parameters (dict[str, Any]): Solver Options

    """
    DB = Database.get_instance()
    bodies_dicts: list[GenuSurface] = []
    if solver_parameters["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    disturbances: list[Disturbance] = state.disturbances

    from multiprocessing import Pool

    def run() -> None:
        if gnvp_version == 3:
            num_processes = CPU_TO_USE
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
                    state.trim["U"],
                    state.trim["AoA"],
                    state.environment,
                    surfaces,
                    bodies_dicts,
                    dst,
                    "Dynamics",
                    gnvp_version,
                    solver_parameters,
                )
                for dst in disturbances
            ]

            _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)

    folders: list[str] = [disturbance_to_directory(dst) for dst in disturbances]
    GENUDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )
    CASEDIRS: list[str] = [os.path.join(GENUDIR, "Dynamics", folder) for folder in folders]

    refresh_progress: float = 2
    job = Thread(target=run)
    job_monitor = Thread(
        target=parallel_monitor,
        kwargs={
            "CASEDIRS": CASEDIRS,
            "variables": [
                f"{dst.var} - {dst.amplitude:.4f}" if dst.amplitude else f"{dst.var}" for dst in disturbances
            ],
            "max_iter": maxiter,
            "refresh_progress": refresh_progress,
            "gnvp_version": gnvp_version,
        },
    )

    # Start the job
    job.start()
    job_monitor.start()

    # Wait for the job to finish
    job.join()
    job_monitor.join()


def process_gnvp3_dynamics(plane: Airplane, state: State) -> DataFrame:
    return process_gnvp_dynamics(plane, state, 3)


def process_gnvp7_dynamics(plane: Airplane, state: State) -> DataFrame:
    return process_gnvp_dynamics(plane, state, 7)


def process_gnvp_dynamics(
    plane: Airplane,
    state: State,
    gnvp_version: int,
    default_name_to_use: str | None = None,
) -> DataFrame:
    """Process the pertrubation results from the GNVP solver

    Args:
        plane (Airplane): Airplane Object
        state (State): Plane State to load results to
        gnvp_version (int): GenuVP version

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation

    """
    DB = Database.get_instance()
    forces: DataFrame = forces_to_pertrubation_results(
        plane,
        state,
        gnvp_version,
        default_name_to_use=default_name_to_use,
    )
    state.set_pertrubation_results(forces)
    state.stability_fd()
    # Save the state
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )
    state.save(CASEDIR)
    if plane.name not in DB.vehicles_db.states:
        DB.vehicles_db.states[plane.name] = {}
    DB.vehicles_db.states[plane.name][state.name] = state
    return forces
