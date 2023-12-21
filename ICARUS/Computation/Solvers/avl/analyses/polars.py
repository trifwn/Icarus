import logging
import os
from typing import Any

from pandas import DataFrame

from ICARUS.Computation.Solvers.AVL.files.input import make_input_files
from ICARUS.Computation.Solvers.AVL.files.polars import case_def
from ICARUS.Computation.Solvers.AVL.files.polars import case_run
from ICARUS.Computation.Solvers.AVL.files.polars import case_setup
from ICARUS.Computation.Solvers.AVL.post_process.post import collect_avl_polar_forces
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import DB3D
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def avl_angle_run(
    plane: Airplane,
    state: State,
    solver2D: str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any] = {},
) -> None:
    planedir = os.path.join(DB3D, plane.directory, "AVL")
    make_input_files(planedir, plane, state, solver2D)
    case_def(plane, angles)
    case_setup(plane)
    case_run(plane, angles)


def process_avl_angles_run(plane: Airplane, state: State, angles: FloatArray | list[float]) -> DataFrame:
    """Procces the results of the GNVP3 AoA Analysis and
    return the forces calculated in a DataFrame

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        genu_version: GNVP Version

    Returns:
        DataFrame: Forces Calculated
    """
    HOMEDIR: str = DB.HOMEDIR
    CASEDIR: str = os.path.join(DB3D, plane.directory)

    forces: DataFrame = collect_avl_polar_forces(plane=plane, state=state, angles=angles)
    plane.save()
    state.save(CASEDIR)

    logging.info("Adding Results to Database")
    # Add Plane to Database
    file_plane: str = os.path.join(DB3D, plane.directory, f"{plane.name}.json")
    _ = DB.vehicles_db.load_plane(name=plane.name, file=file_plane)

    # Add Results to Database
    DB.vehicles_db.load_avl_data(
        plane=plane,
        state=state,
        vehicle_folder=plane.directory,
    )

    return forces
