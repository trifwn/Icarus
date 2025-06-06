import logging
import os
from typing import Any
from typing import Literal

from pandas import DataFrame

from ICARUS.computation.solvers.AVL import collect_avl_polar_forces
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

from ..files.input import make_input_files
from ..files.polars import case_run
from ..files.polars import case_setup
from ..files.polars import run_file


def avl_polars(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any] = {"use_avl_control": False},
) -> None:
    DB = Database.get_instance()
    case_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )

    run_file(case_directory, plane, state, angles)
    make_input_files(case_directory, plane, state, solver2D, solver_options)
    case_setup(case_directory, plane, state)
    case_run(case_directory, plane, angles)
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
        gnvp_version: GNVP Version

    Returns:
        DataFrame: Forces Calculated

    """
    DB = Database.get_instance()

    case_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )

    forces: DataFrame = collect_avl_polar_forces(
        directory=case_directory,
        plane=plane,
        state=state,
        angles=angles,
    )
    filename = os.path.join(case_directory, "forces.avl")
    forces.to_csv(filename, index=False, float_format="%.10f")

    plane.save()

    try:
        state.add_polar(
            polar=forces,
            polar_prefix="AVL",
            is_dimensional=True,
        )

    except Exception as e:
        raise (e)

    finally:
        state.save(case_directory)
        logging.info("Adding Results to Database")
        # Add Plane to Database
        file_plane: str = os.path.join(case_directory, f"{plane.name}.json")
        _ = DB.load_vehicle(name=plane.name, file=file_plane)

        # Add Results to Database
        DB.load_vehicle_solver_data(
            vehicle=plane,
            state=state,
            folder=case_directory,
            solver="AVL",
        )
    return forces
