import os
from threading import Thread
from typing import Any

from pandas import DataFrame
from tqdm.auto import tqdm

from ICARUS import CPU_TO_USE
from ICARUS.computation.solvers.GenuVP.analyses.monitor_progress import parallel_monitor
from ICARUS.computation.solvers.GenuVP.analyses.monitor_progress import serial_monitor
from ICARUS.computation.solvers.GenuVP.files.gnvp3_interface import run_gnvp3_case
from ICARUS.computation.solvers.GenuVP.files.gnvp7_interface import run_gnvp7_case
from ICARUS.computation.solvers.GenuVP.post_process.forces import (
    forces_to_pertrubation_results,
)
from ICARUS.computation.solvers.GenuVP.post_process.forces import rotate_gnvp_forces
from ICARUS.computation.solvers.GenuVP.utils.genu_movement import Movement
from ICARUS.computation.solvers.GenuVP.utils.genu_movement import define_movements
from ICARUS.computation.solvers.GenuVP.utils.genu_parameters import GenuParameters
from ICARUS.computation.solvers.GenuVP.utils.genu_surface import GenuSurface
from ICARUS.core.struct import Struct
from ICARUS.database import DB
from ICARUS.database.utils import disturbance_to_case
from ICARUS.environment.definition import Environment
from ICARUS.flight_dynamics.disturbances import Disturbance
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.surface import WingSurface


def gnvp_disturbance_case(
    plane: Airplane,
    solver2D: str,
    maxiter: int,
    timestep: float,
    u_freestream: float,
    angle: float,
    environment: Environment,
    surfaces: list[WingSurface],
    bodies_dicts: list[GenuSurface],
    dst: Disturbance,
    analysis: str,
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> str:
    """
    Run a single disturbance simulation in GNVP3

    Args:
        plane (Airplane): Plane Object
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
        solver_options (dict[str, Any] | Struct): Solver Options

    Returns:
        str: Case Done Message
    """
    HOMEDIR: str = DB.HOMEDIR
    PLANEDIR: str = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver=f"GenuVP{genu_version}",
    )
    airfoils: list[str] = plane.airfoils

    movements: list[list[Movement]] = define_movements(
        surfaces,
        plane.CG,
        plane.orientation,
        [dst],
    )

    folder: str = disturbance_to_case(dst)
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
        solver_options,
    )

    if genu_version == 7:
        run = run_gnvp7_case
    else:
        run = run_gnvp3_case

    run(
        CASEDIR,
        HOMEDIR,
        movements,
        bodies_dicts,
        params,
        airfoils,
        solver2D,
    )

    return f"Case {dst.var} : {dst.amplitude} Done"


def run_gnvp3_pertrubation_serial(*args: Any, **kwars: Any) -> None:
    run_pertrubation_serial(genu_version=3, *args, **kwars)  # type: ignore


def run_gnvp7_pertrubation_serial(*args: Any, **kwars: Any) -> None:
    run_pertrubation_serial(genu_version=7, *args, **kwars)  # type: ignore


def run_gnvp3_pertrubation_parallel(*args: Any, **kwars: Any) -> None:
    run_pertrubation_parallel(genu_version=3, *args, **kwars)  # type: ignore


def run_gnvp7_pertrubation_parallel(*args: Any, **kwars: Any) -> None:
    run_pertrubation_parallel(genu_version=7, *args, **kwars)  # type: ignore


def run_pertrubation_serial(
    plane: Airplane,
    state: State,
    solver2D: str,
    maxiter: int,
    timestep: float,
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is serial.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
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
                "plane": plane,
                "solver2D": solver2D,
                "maxiter": maxiter,
                "timestep": timestep,
                "u_freestream": state.u_freestream,
                "angle": state.trim["U"],
                "environment": state.environment,
                "surfaces": surfaces,
                "bodies_dicts": bodies_dicts,
                "dst": dst,
                "analysis": "Dynamics",
                "genu_version": genu_version,
                "solver_options": solver_options,
            },
        )
        pbar = tqdm(
            total=maxiter,
            desc=f"DST:{dst.var} - {dst.amplitude}",
            position=i,
            leave=True,
            colour="RED",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        progress_bars.append(pbar)
        folder: str = disturbance_to_case(dst)
        PLANEDIR: str = DB.vehicles_db.get_case_directory(
            airplane=plane,
            solver=f"GenuVP{genu_version}",
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
                "genu_version": genu_version,
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
    solver2D: str,
    maxiter: int,
    timestep: float,
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is parallel.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        solver2D (str): Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    disturbances: list[Disturbance] = state.disturbances

    from multiprocessing import Pool

    def run() -> None:
        if genu_version == 3:
            num_processes = CPU_TO_USE
        else:
            num_processes = int(CPU_TO_USE / 3)
        with Pool(num_processes) as pool:
            args_list = [
                (
                    plane,
                    solver2D,
                    maxiter,
                    timestep,
                    state.u_freestream,
                    state.trim["U"],
                    state.environment,
                    surfaces,
                    bodies_dicts,
                    dst,
                    "Dynamics",
                    genu_version,
                    solver_options,
                )
                for dst in disturbances
            ]

            _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)

    folders: list[str] = [disturbance_to_case(dst) for dst in disturbances]
    GENUDIR: str = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver=f"GenuVP{genu_version}",
    )
    CASEDIRS: list[str] = [os.path.join(GENUDIR, "Dynamics", folder) for folder in folders]

    refresh_progress: float = 2
    job = Thread(target=run)
    job_monitor = Thread(
        target=parallel_monitor,
        kwargs={
            "CASEDIRS": CASEDIRS,
            "variables": [f"{dst.var} - {dst.amplitude}" for dst in disturbances],
            "max_iter": maxiter,
            "refresh_progress": refresh_progress,
            "genu_version": genu_version,
        },
    )

    # Start the job
    job.start()
    job_monitor.start()

    # Wait for the job to finish
    job.join()
    job_monitor.join()


def run_gnvp3_sensitivity_serial(*args: Any, **kwars: Any) -> None:
    sensitivity_serial(genu_version=3, *args, **kwars)  # type: ignore


def run_gnvp7_sensitivity_serial(*args: Any, **kwars: Any) -> None:
    sensitivity_serial(genu_version=7, *args, **kwars)  # type: ignore


def run_gnvp3_sensitivity_parallel(*args: Any, **kwars: Any) -> None:
    sensitivity_parallel(genu_version=3, *args, **kwars)  # type: ignore


def run_gnvp7_sensitivity_parallel(*args: Any, **kwars: Any) -> None:
    sensitivity_parallel(genu_version=7, *args, **kwars)  # type: ignore


def sensitivity_serial(
    plane: Airplane,
    state: State,
    var: str,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angle: float,
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is serial.

    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        var (str): Variable to be perturbed
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        angle_of_attack (float): Angle of attack in degrees
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    for dst in state.sensitivities[var]:
        msg: str = gnvp_disturbance_case(
            plane,
            solver2D,
            maxiter,
            timestep,
            state.u_freestream,
            angle,
            state.environment,
            surfaces,
            bodies_dicts,
            dst,
            "Sensitivity",
            genu_version,
            solver_options,
        )
        print(msg)


def sensitivity_parallel(
    plane: Airplane,
    state: State,
    var: str,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angle: float,
    genu_version: int,
    solver_options: dict[str, Any] | Struct,
) -> None:
    """
    For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is parallel.

    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        var (str): Variable to be perturbed
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        angle_of_attack (float): Angle of attack in degrees
        solver_options (dict[str, Any] | Struct): Solver Options
    """
    bodies_dicts: list[GenuSurface] = []
    if solver_options["Split_Symmetric_Bodies"]:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    disturbances: list[Disturbance] = state.sensitivities[var]

    from multiprocessing import Pool

    if genu_version == 3:
        num_processes = CPU_TO_USE
    else:
        num_processes = int(CPU_TO_USE / 3)
    with Pool(num_processes) as pool:
        args_list = [
            (
                plane,
                solver2D,
                maxiter,
                timestep,
                state.u_freestream,
                angle,
                state.environment,
                surfaces,
                bodies_dicts,
                dst,
                f"Sensitivity_{dst.var}",
                genu_version,
                solver_options,
            )
            for dst in disturbances
        ]

        _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)


def proccess_pertrubation_res_3(plane: Airplane, state: State) -> DataFrame:
    return proccess_pertrubation_res(plane, state, 3)


def proccess_pertrubation_res_7(plane: Airplane, state: State) -> DataFrame:
    return proccess_pertrubation_res(plane, state, 7)


def proccess_pertrubation_res(plane: Airplane, state: State, gnvp_version: int) -> DataFrame:
    """
    Process the pertrubation results from the GNVP solver

    Args:
        plane (Airplane): Airplane Object
        state (State): Plane State to load results to
        genu_version (int): GenuVP version

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation
    """
    HOMEDIR: str = DB.HOMEDIR
    DYNDIR: str = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver=f"GenuVP{gnvp_version}",
        case="Dynamics",
    )
    forces: DataFrame = forces_to_pertrubation_results(DYNDIR, HOMEDIR, state, gnvp_version)
    forces = rotate_gnvp_forces(forces, forces["AoA"], gnvp_version)

    state.set_pertrubation_results(forces)
    state.stability_fd()
    # Save the state
    PLANEDIR = DB.vehicles_db.get_plane_directory(
        plane=plane,
    )
    state.save(PLANEDIR)
    DB.vehicles_db.states[plane.name] = state

    return forces


# def processGNVPsensitivity(plane):
#     HOMEDIR = DB.HOMEDIR
#     DYNDIR = os.path.join(DB.vehiclesDB.DATADIR, plane.CASEDIR, "Dynamics")
#     forces = forces2pertrubRes(DYNDIR, HOMEDIR)
#     # rotatedforces = rotateForces(forces, forces["AoA"])
#     return forces #rotatedforces
