import logging
import os

from pandas import DataFrame

from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

from .. import AVLParameters
from ..files.cases import AVLRunSetup
from ..files.cases import avl_run_cases
from ..files.input import make_input_files
from ..post_process import collect_avl_polar_forces

AVL_LOGGER = logging.getLogger("ICARUS.solvers.GenuVP")


def avl_polars(
    plane: Airplane,
    state: State,
    angles: FloatArray | list[float],
    solver_parameters: AVLParameters = AVLParameters(),
) -> None:
    DB = Database.get_instance()
    case_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )

    make_input_files(case_directory, plane, state, solver_parameters)
    run_setup = AVLRunSetup.aseq(
        name=f"{plane.name}_polars.run",
        state=state,
        angles=angles,
    )
    avl_run_cases(case_directory, plane, run_setup)
    polar_df = process_avl_polars(plane, state, run_setup)
    state.add_polar(polar_df, polar_prefix="AVL", is_dimensional=True)


def process_avl_polars(
    plane: Airplane,
    state: State,
    avl_run: AVLRunSetup,
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
        avl_run=avl_run,
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
        AVL_LOGGER.info("Adding Results to Database")
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
