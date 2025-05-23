import logging
import os
from typing import Any
from typing import Literal

from pandas import DataFrame

from ICARUS.computation.solvers.AVL import collect_avl_polar_forces
from ICARUS.computation.solvers.AVL.files.input import make_input_files
from ICARUS.computation.solvers.AVL.files.polars import case_def
from ICARUS.computation.solvers.AVL.files.polars import case_run
from ICARUS.computation.solvers.AVL.files.polars import case_setup
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane


def avl_polars(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any] = {"use_avl_control": False},
) -> None:
    DB = Database.get_instance()
    PLANEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )
    os.makedirs(PLANEDIR, exist_ok=True)
    case_def(plane, state, angles)
    make_input_files(PLANEDIR, plane, state, solver2D, solver_options)
    case_setup(plane, state)
    case_run(plane, state, angles)
    polar_df = process_avl_polars(plane, state, angles)
    state.add_polar(polar_df, polar_prefix="AVL", is_dimensional=True, verbose=False)


def process_avl_polars(
    plane: Airplane,
    state: State,
    angles: FloatArray | list[float],
) -> DataFrame:
    """Procces the results of the GNVP3 AoA Analysis and
    return the forces calculated in a DataFrame

    Args:
        plane (Airplane): Plane Object
        state (State): State of the Airplane
        genu_version: GNVP Version

    Returns:
        DataFrame: Forces Calculated

    """
    DB = Database.get_instance()
    forces: DataFrame = collect_avl_polar_forces(
        plane=plane,
        state=state,
        angles=angles,
    )
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )
    filename = os.path.join(CASEDIR, "forces.avl")
    forces.to_csv(filename, index=False, float_format="%.10f")

    plane.save()
    state.add_polar(
        polar=forces,
        polar_prefix="AVL",
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
        solver="AVL",
    )
    return forces
