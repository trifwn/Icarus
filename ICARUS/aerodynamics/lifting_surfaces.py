import logging
import os
from typing import Any

import pandas as pd
from numpy import ndarray

from ICARUS.aerodynamics import assemble_matrix
from ICARUS.aerodynamics.biot_savart import symm_wing_panels
from ICARUS.aerodynamics.biot_savart import voring
from ICARUS.aerodynamics.wing_lspt import LSPT_Plane
from ICARUS.core.types import FloatArray
from ICARUS.database import DB
from ICARUS.database import DB3D
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


def run_lstp_angles(
    plane: Airplane,
    state: State,
    solver2D: str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any],
) -> None:
    """
    Function to run the wing LLT solver

    Args:
        plane (Airplane): Airplane Object
        options (dict[str, Any]): Options
        solver_options (dict[str, Any]): Solver Options
    """
    LSPTDIR = DB.vehicles_db.get_case_directory(
        airplane=plane,
        solver="LSPT",
    )

    os.makedirs(LSPTDIR, exist_ok=True)
    # Generate the wing LLT solver
    wing = LSPT_Plane(
        plane=plane,
    )

    # Run the solver
    import numpy as np

    if not isinstance(angles, ndarray):
        angles = np.array(angles)

    df: pd.DataFrame = wing.aseq(
        angles=angles,
        state=state,
    )

    # Save the results
    save_results(plane, state, df)


def save_results(
    plane: Airplane,
    state: State,
    df: pd.DataFrame,
) -> None:
    plane_dir: str = os.path.join(DB.vehicles_db.DATADIR, plane.name)
    try:
        os.chdir(plane_dir)
    except FileNotFoundError:
        os.makedirs(plane_dir, exist_ok=True)
        os.chdir(plane_dir)

    # Save the Forces
    df.to_csv(f"forces.lspt", index=False)
    os.chdir(DB.HOMEDIR)

    # Save the Plane
    plane.save()

    logging.info("Adding Results to Database")
    # Add plane to database
    file_plane: str = os.path.join(DB3D, plane.name, f"{plane.name}.json")
    _ = DB.vehicles_db.load_plane(name=plane.name, file=file_plane)

    # Add Forces to Database
    DB.vehicles_db.load_lspt_data(
        plane=plane,
        state=state,
        vehicle_folder=plane.directory,
    )
