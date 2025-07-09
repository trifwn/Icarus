from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Literal
from typing import Sequence

from pandas import DataFrame

from ICARUS.database import Database
from ICARUS.database import disturbance_to_directory

from .. import GenuVP3Parameters
from .. import GenuVP7Parameters
from ..files import gnvp3_case
from ..files import gnvp7_case
from ..post_process import forces_to_pertrubation_results
from ..utils import GenuCaseParams
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from ..utils import define_movements

from ICARUS.flight_dynamics import Disturbance

if TYPE_CHECKING:
    from ICARUS.environment import Environment
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane
    from ICARUS.vehicle import WingSurface


def gnvp_disturbance_case(
    DB: Database,
    plane: Airplane,
    state: State,
    u_freestream: float,
    angle: float,
    environment: Environment,
    surfaces: Sequence[WingSurface],
    bodies_dicts: list[GenuSurface],
    dst: Disturbance,
    analysis: str,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> None:
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
    if isinstance(solver_parameters, GenuVP3Parameters):
        gnvp_version: Literal[3, 7] = 3
    elif isinstance(solver_parameters, GenuVP7Parameters):
        gnvp_version = 7
    else:
        raise ValueError(
            "solver_parameters must be an instance of GenuVP3Parameters or GenuVP7Parameters",
        )

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

    params: GenuCaseParams = GenuCaseParams(
        bodies_dicts,
        plane,
        u_freestream,
        angle,
        environment,
        solver_parameters,
    )

    if gnvp_version == 7:
        gnvp7_case(
            CASEDIR,
            movements,
            bodies_dicts,
            params,
        )
    else:
        gnvp3_case(
            CASEDIR,
            movements,
            bodies_dicts,
            params,
        )


def gnvp_stability(
    plane: Airplane,
    state: State,
    disturbances: Sequence[Disturbance] | Disturbance,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> None:
    """For each pertrubation in the plane object, run a simulation in GNVP3.
    Can be used mainly for a pertrubation analysis. This analysis is serial.

    Args:
        plane (Airplane): Airplane Object
        state (State): Dynamic State of the airplane
        disturbances (Sequence[Disturbance] | Disturbance): Disturbances to be run
        solver_parameters (GenuVP3Parameters | GenuVP7Parameters): Solver Parameters

    """
    DB = Database.get_instance()
    bodies_dicts: list[GenuSurface] = []
    if solver_parameters.Split_Symmetric_Bodies:
        surfaces: list[WingSurface] = plane.get_seperate_surfaces()
    else:
        surfaces = plane.surfaces

    for i, surface in enumerate(surfaces):
        genu_surf = GenuSurface(surface, i)
        bodies_dicts.append(genu_surf)

    if isinstance(disturbances, Disturbance):
        disturbances_list: list[Disturbance] = [disturbances]
    else:
        disturbances_list = list(disturbances)

    for dst in disturbances_list:
        gnvp_disturbance_case(
            DB=DB,
            plane=plane,
            state=state,
            u_freestream=state.trim["U"],
            angle=state.trim["AoA"],
            environment=state.environment,
            surfaces=surfaces,
            bodies_dicts=bodies_dicts,
            dst=dst,
            analysis="Dynamics",
            solver_parameters=solver_parameters,
        )


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
