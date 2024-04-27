import logging
import os
from typing import Any
from typing import Literal

from pandas import DataFrame

from ICARUS.computation.solvers.AVL.files.input import make_input_files
from ICARUS.computation.solvers.AVL.files.polars import case_def
from ICARUS.computation.solvers.AVL.files.polars import case_run
from ICARUS.computation.solvers.AVL.files.polars import case_setup
from ICARUS.computation.solvers.AVL.post_process.post import collect_avl_polar_forces
from ICARUS.core.types import FloatArray
from ICARUS.database import DB
from ICARUS.database import DB3D
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


def avl_angle_run(
    plane: Airplane,
    state: State,
    solver2D: Literal['Xfoil', 'Foil2Wake', 'OpenFoam'] | str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any] = {"use_avl_control": False},
) -> None:
    PLANEDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="AVL",
    )
    os.makedirs(PLANEDIR, exist_ok=True)
    case_def(plane, state, angles)
    make_input_files(PLANEDIR, plane, state, solver2D, solver_options)
    case_setup(plane)
    case_run(plane, angles)
    _ = process_avl_angles_run(plane, state, angles)


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
    forces: DataFrame = collect_avl_polar_forces(plane=plane, state=state, angles=angles)
    plane.save()
    CASEDIR = DB.vehicles_db.get_plane_directory(
        plane=plane,
    )
    state.save(CASEDIR)

    logging.info("Adding Results to Database")
    # Add Plane to Database
    file_plane: str = os.path.join(CASEDIR, f"{plane.name}.json")
    _ = DB.vehicles_db.load_plane(name=plane.name, file=file_plane)

    # Add Results to Database
    DB.vehicles_db.load_avl_data(
        plane=plane,
        state=state,
        vehicle_folder=plane.directory,
    )

    return forces
