from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from ICARUS import CPU_TO_USE
from ICARUS.core.base_types import Struct
from ICARUS.database import Database

from ..utils import GenuSurface
from .pertrubations import gnvp_disturbance_case

if TYPE_CHECKING:
    from ICARUS.flight_dynamics import Disturbance
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface


def gnvp_sensitivities(
    *args: Any,
    gnvp_version,
    parallel,
    **kwars: Any,
) -> None:
    """Wrapper function for sensitivities_serial and sensitivities_parallel.

    Args:
        gnvp_version (int, optional): Genu Version. Defaults to 3.
        parallel (bool, optional): Run in parallel. Defaults to False.
        *args (Any): Arguments for the function.
        **kwars (Any): Keyword arguments for the function.

    """
    if gnvp_version == 3:
        if parallel:
            sensitivities_parallel(*args, **kwars)
        else:
            sensitivities_serial(*args, **kwars)  # type: ignore
    elif gnvp_version == 7:
        if parallel:
            sensitivities_parallel(*args, **kwars)  # type: ignore
        else:
            sensitivities_serial(*args, **kwars)
    else:
        raise ValueError("Genu version must be either 3 or 7.")


def gnvp3_sensitivities_serial(*args: Any, **kwars: Any) -> None:
    sensitivities_serial(gnvp_version=3, *args, **kwars)  # type: ignore


def gnvp7_sensitivities_serial(*args: Any, **kwars: Any) -> None:
    sensitivities_serial(gnvp_version=7, *args, **kwars)  # type: ignore


def gnvp3_sensitivities_parallel(*args: Any, **kwars: Any) -> None:
    sensitivities_parallel(gnvp_version=3, *args, **kwars)  # type: ignore


def gnvp7_sensitivities_parallel(*args: Any, **kwars: Any) -> None:
    sensitivities_parallel(gnvp_version=7, *args, **kwars)  # type: ignore


def sensitivities_serial(
    plane: Airplane,
    state: State,
    var: str,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angle: float,
    gnvp_version: int,
    solver_parameters: dict[str, Any] | Struct,
) -> None:
    """For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is serial.

    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        var (str): Variable to be perturbed
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        angle_of_attack (float): Angle of attack in degrees
        solver_parameters (dict[str, Any] | Struct): Solver Options

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

    for dst in state.sensitivities[var]:
        msg: str = gnvp_disturbance_case(
            DB,
            plane,
            state,
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
            gnvp_version,
            solver_parameters,
        )
        print(msg)


def sensitivities_parallel(
    plane: Airplane,
    state: State,
    var: str,
    solver2D: str,
    maxiter: int,
    timestep: float,
    angle: float,
    gnvp_version: int,
    solver_parameters: dict[str, Any] | Struct,
) -> None:
    """For each pertrubation in the sensitivity attribute of the dynamic airplane
    object, run a simulation in GNVP3. Can be used mainly for a sensitivity
    analysis. This analysis is parallel.

    Args:
        plane (Dynamic_Airplane): Dynamic Airplane Object
        var (str): Variable to be perturbed
        solver2D (str): 2D Solver to be used for foil data
        maxiter (int): Max Iterations
        timestep (float): Timestep for the simulation
        angle_of_attack (float): Angle of attack in degrees
        solver_parameters (dict[str, Any] | Struct): Solver Options

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

    disturbances: list[Disturbance] = state.sensitivities[var]

    from multiprocessing import Pool

    if gnvp_version == 3:
        num_processes = CPU_TO_USE
    else:
        num_processes = int(CPU_TO_USE / 3)
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
                surfaces,
                bodies_dicts,
                dst,
                f"Sensitivity_{dst.var}",
                gnvp_version,
                solver_parameters,
            )
            for dst in disturbances
        ]

        _: list[str] = pool.starmap(gnvp_disturbance_case, args_list)
