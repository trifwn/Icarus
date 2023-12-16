from typing import Any

import numpy as np
from pandas import DataFrame

from ICARUS.Computation.Solvers.AVL.analyses.polars import avl_angle_run
from ICARUS.Computation.Solvers.AVL.analyses.polars import process_avl_angles_run
from ICARUS.Computation.Solvers.AVL.files.dynamics import finite_difs
from ICARUS.Computation.Solvers.AVL.files.dynamics import implicit_eigs
from ICARUS.Computation.Solvers.AVL.post_process.post import finite_difs_post
from ICARUS.Computation.Solvers.AVL.post_process.post import implicit_dynamics_post
from ICARUS.Database import DB
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def avl_dynamic_analysis_implicit(
    plane: Airplane,
    state: State,
    solver2D: str,
    solver_options: dict[str, Any] = {},
) -> None:
    implicit_eigs(plane=plane, state=state, solver2D=solver2D)


def avl_dynamic_analysis_fd(plane: Airplane, state: State, solver2D: str, solver_options: dict[str, Any] = {}) -> None:
    if state.trim == {}:
        angles = np.linspace(-10, 10, 21)
        avl_angle_run(plane, state, solver2D=solver2D, angles=angles)
        polar_df = process_avl_angles_run(plane, state, angles)
        state.add_polar(polar_df, polar_prefix="AVL", is_dimensional=True)

    finite_difs(plane=plane, state=state, solver2D=solver2D)


def process_avl_fd_res(plane: Airplane, state: State) -> DataFrame:
    """
    Process the pertrubation results from the AVL solver

    Args:
        plane (Airplane): Airplane Object
        state (State): Plane State to load results to
        genu_version (int): GenuVP version

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation
    """
    forces: DataFrame = finite_difs_post(plane, state)

    state.set_pertrubation_results(forces)
    state.stability_fd()
    DB.vehicles_db.states[plane.name] = state

    return forces


def process_avl_impl_res(plane: Airplane, state: State) -> tuple[list[complex], list[complex]]:
    impl_long, impl_late = implicit_dynamics_post(plane, state)
    return impl_long, impl_late
