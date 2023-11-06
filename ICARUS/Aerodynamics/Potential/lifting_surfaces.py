import os
from re import L
from typing import Any
from typing import Callable

import pandas as pd
from regex import D

from ICARUS.Aerodynamics.Potential.vorticity import symm_wing_panels
from ICARUS.Aerodynamics.Potential.vorticity import voring
from ICARUS.Aerodynamics.Potential.wing_lspt import Wing_LSPT
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import DB3D
from ICARUS.Environment.definition import Environment
from ICARUS.Vehicle.plane import Airplane


def run_lstp_angles(
    plane: Airplane,
    environment: Environment,
    solver2D: str,
    u_freestream: float,
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

    # Generate the wing LLT solver
    wing = Wing_LSPT(
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
    save_results(plane, df)


def save_results(
    plane: Airplane,
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

    print("Adding Results to Database")
    # Add plane to database
    file_plane: str = os.path.join(DB3D, plane.name, f"{plane.name}.json")
    _ = DB.vehicles_db.load_plane_from_file(name=plane.name, file=file_plane)

    # Add Forces to Database
    file_gnvp: str = os.path.join(DB3D, plane.name, f"forces.lspt")
    DB.vehicles_db.load_lspt_forces(planename=plane.name, file=file_gnvp)
