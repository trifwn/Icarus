from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
from pandas import DataFrame

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.database import angle_to_directory
from ICARUS.vehicle.surface import WingSurface

from .. import GenuVP3Parameters
from .. import GenuVP7Parameters
from ..files import gnvp3_case
from ..files import gnvp7_case
from ..post_process import log_forces
from ..utils import GenuCaseParams
from ..utils import GenuSurface
from ..utils import GNVP_Movement
from ..utils import define_global_movements

if TYPE_CHECKING:
    from ICARUS.environment import Environment
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane


class StopRunningThreadError(Exception):
    pass


GNVP_LOGGER = logging.getLogger("ICARUS.solvers.GenuVP")


def gnvp_aoa_case(
    DB: Database,
    plane: Airplane,
    state: State,
    u_freestream: float,
    angle: float,
    environment: Environment,
    movements: list[list[GNVP_Movement]],
    bodies_dicts: list[GenuSurface],
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
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
        solver_parameters (dict[str, Any] | Struct): Solver Options

    """
    if isinstance(solver_parameters, GenuVP3Parameters):
        gnvp_version: Literal[3, 7] = 3
    elif isinstance(solver_parameters, GenuVP7Parameters):
        gnvp_version = 7
    else:
        raise TypeError(
            "solver_parameters must be of type GenuVP3Parameters or GenuVP7Parameters",
        )

    PLANEDIR: str = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
    )

    case_directory: str = os.path.join(PLANEDIR, angle_to_directory(angle))
    os.makedirs(case_directory, exist_ok=True)

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
            case_directory=case_directory,
            movements=movements,
            genu_bodies=bodies_dicts,
            params=params,
        )
    else:
        gnvp3_case(
            case_directory=case_directory,
            movements=movements,
            genu_bodies=bodies_dicts,
            params=params,
        )


def gnvp_aseq(
    plane: Airplane,
    state: State,
    angles: list[float] | FloatArray,
    solver_parameters: GenuVP3Parameters | GenuVP7Parameters,
) -> None:
    """Run Multiple Angles Simulation in GNVP3-7 Serially

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        solver2D (str): Name of 2D Solver to be used for the 2d polars
        maxiter (int): Maxiteration for each case
        timestep (float): Timestep for simulations
        angles (list[float]): List of angles to run
        gnvp_version (int): Version of GenuVP solver
        solver_parameters (GenuVP3Parameters | GenuVP7Parameters): Solver Options

    """
    bodies_dicts: list[GenuSurface] = []
    if solver_parameters.Split_Symmetric_Bodies:
        surfaces: list[tuple[int, WingSurface]] = plane.split_wing_segments()
    else:
        surfaces = plane.wing_segments

    for i, surface in surfaces:
        gen_surf: GenuSurface = GenuSurface(surface, i)
        bodies_dicts.append(gen_surf)

    global_movements: list[GNVP_Movement] = define_global_movements(
        plane.CG,
        plane.orientation,
    )
    movements: list[list[GNVP_Movement]] = [global_movements for _ in bodies_dicts]

    DB = Database.get_instance()

    if isinstance(angles, float):
        angles_list: list[float] = [angles]
    elif isinstance(angles, np.ndarray):
        angles_list = angles.tolist()
    elif isinstance(angles, list):
        angles_list = angles
    else:
        raise TypeError(
            "angles must be a float, np.ndarray, or list of floats",
        )

    for i, angle in enumerate(angles_list):
        gnvp_aoa_case(
            DB=DB,
            plane=plane,
            state=state,
            u_freestream=state.u_freestream,
            angle=float(angle),
            environment=state.environment,
            movements=movements,
            bodies_dicts=bodies_dicts,
            solver_parameters=solver_parameters,
        )

    GenuSurface.airfoil_names = {}
    GenuSurface.surf_names = {}


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

    GNVP_LOGGER.info("Adding Results to Database")
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
