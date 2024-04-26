from typing import Any
from typing import Literal

import numpy as np
from pandas import DataFrame

from ICARUS.computation.solvers.AVL.analyses.polars import avl_angle_run
from ICARUS.computation.solvers.AVL.analyses.polars import process_avl_angles_run
from ICARUS.computation.solvers.AVL.files.dynamics import finite_difs
from ICARUS.computation.solvers.AVL.files.dynamics import implicit_eigs
from ICARUS.computation.solvers.AVL.post_process.post import finite_difs_post
from ICARUS.computation.solvers.AVL.post_process.post import implicit_dynamics_post
from ICARUS.database import DB
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


def avl_dynamic_analysis_implicit(
    plane: Airplane,
    state: State,
    solver2D: Literal['Xfoil', 'Foil2Wake', 'OpenFoam'] | str = 'Xfoil',
    solver_options: dict[str, Any] = {},
) -> None:
    implicit_eigs(plane=plane, state=state, solver2D=solver2D)


def avl_dynamic_analysis_fd(
    plane: Airplane,
    state: State,
    solver2D: Literal['Xfoil', 'Foil2Wake', 'OpenFoam'] | str = 'Xfoil',
    solver_options: dict[str, Any] = {},
) -> None:
    if state.trim == {}:
        print("Trimming the plane")
        aoa_min = -8
        aoa_max = 15
        num_aoa = (aoa_max - aoa_min) * 2 + 1
        # angles = np.linspace(aoa_min, aoa_max, num_aoa)
        angles = np.linspace(-10, 10, 21)

        avl_angle_run(plane=plane, state=state, solver2D=solver2D, angles=angles, solver_options=solver_options)
        polar_df = process_avl_angles_run(plane, state, angles)
        state.add_polar(polar_df, polar_prefix="AVL", is_dimensional=True, verbose=False)

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
