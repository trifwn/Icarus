import os
from typing import Any
from typing import Callable

import pandas as pd
from regex import D

from ICARUS.Aerodynamics.Potential.vorticity import symm_wing_panels
from ICARUS.Aerodynamics.Potential.vorticity import voring
from ICARUS.Aerodynamics.Potential.wing_lspt import Wing_LSPT
from ICARUS.Core.types import FloatArray
from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import Environment
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
        solve_fun: Callable[..., tuple[FloatArray, FloatArray]] = symm_wing_panels
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
