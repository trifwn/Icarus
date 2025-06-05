from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray

from ICARUS.aero import LSPT_Plane
from ICARUS.aero.aerodynamic_results import AerodynamicResults
from ICARUS.database import Database

if TYPE_CHECKING:
    from ICARUS.core.types import FloatArray
    from ICARUS.flight_dynamics import State
    from ICARUS.vehicle import Airplane

from .run_vlm import run_vlm_polar_analysis


def lspt_polars(
    plane: Airplane,
    state: State,
    solver2D: str,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any],
) -> None:
    """Function to run the wing LLT solver

    Args:
        plane (Airplane): Airplane Object
        options (dict[str, Any]): Options
        solver_options (dict[str, Any]): Solver Options

    """
    DB = Database.get_instance()
    LSPTDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="LSPT",
    )

    os.makedirs(LSPTDIR, exist_ok=True)
    # Generate the wing LLT solver

    wing = LSPT_Plane(
        plane=plane,
    )

    # Run the solver

    if not isinstance(angles, ndarray):
        angles = np.array(angles)

    results: AerodynamicResults = run_vlm_polar_analysis(
        plane=wing,
        state=state,
        angles=angles,
    )

    # Convert the results to a DataFrame
    df = results.to_polars_dataframe()

    # Save the results
    save_results(plane, state, df)


def save_results(
    plane: Airplane,
    state: State,
    forces_df: pd.DataFrame,
) -> None:
    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="LSPT",
    )
    filename = os.path.join(CASEDIR, "forces.lspt")
    forces_df.to_csv(filename, index=False, float_format="%.10f")

    plane.save()
    state.add_polar(
        polar=forces_df,
        polar_prefix="LSPT Potential",
        is_dimensional=True,
    )

    # Save the Forces
    # Add plane to database
    logging.info("Adding Results to Database")
    file_plane: str = os.path.join(DB.DB3D, plane.name, f"{plane.name}.json")
    _ = DB.load_vehicle(name=plane.name, file=file_plane)

    # Add Forces to Database
    DB.load_vehicle_solver_data(vehicle=plane, state=state, folder=plane.directory, solver="LSPT")
