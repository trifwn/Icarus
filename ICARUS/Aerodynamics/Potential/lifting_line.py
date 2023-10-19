import os
import pandas as pd
from typing import Any

from regex import D
from ICARUS.Core.types import FloatArray
from ICARUS.Enviroment.definition import Environment
from ICARUS.Aerodynamics.Potential.core.wing_lspt import Wing_LSPT
from ICARUS.Aerodynamics.Potential.core.vorticity import symm_wing_panels, voring

from ICARUS.Database.db import DB
from ICARUS.Vehicle.plane import Airplane


def run_lstp_angles(
    plane: Airplane,
    environment: Environment,
    db: DB,
    solver2D: str,
    u_freestream: float,
    angles: FloatArray | list[float],
    solver_options: dict[str, Any],
) -> None:
    """
    Function to run the wing LLT solver

    Args:
        db (DB): Database Object
        plane (Airplane): Airplane Object
        options (dict[str, Any]): Options
        solver_options (dict[str, Any]): Solver Options
    """

    # Generate the wing LLT solver
    wing = Wing_LSPT(
        db=db,
        plane=plane,
        environment=environment,
        alpha=0,
        beta=0,
    )

    if wing.is_symmetric:
        solve_fun = symm_wing_panels
    else:
        solve_fun = voring

    # Run the solver
    df: pd.DataFrame = wing.aseq(
        angles=angles,
        umag=u_freestream,
        solver_fun=solve_fun,
    )

    # Save the results
    save_results(db, plane, df)


def save_results(
    db: DB,
    plane: Airplane,
    df: pd.DataFrame,
) -> None:
    """
    Function to save the results of the LLT solver
    """
    plane_dir: str = os.path.join(db.vehiclesDB.DATADIR, plane.name)
    try:
        os.chdir(plane_dir)
    except FileNotFoundError:
        os.makedirs(plane_dir, exist_ok=True)
        os.chdir(plane_dir)

    df.to_csv(f"forces.lspt", index=False)
    os.chdir(db.HOMEDIR)
