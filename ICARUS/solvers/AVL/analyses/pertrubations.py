from typing import Any
from typing import Literal

import numpy as np
from pandas import DataFrame

from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL.files.dynamics import finite_difs
from ICARUS.solvers.AVL.files.dynamics import implicit_eigs
from ICARUS.vehicle import Airplane


def avl_dynamics_implicit(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
    solver_parameters: dict[str, Any] = {},
) -> None:
    implicit_eigs(plane=plane, state=state, solver2D=solver2D, solver_parameters=solver_parameters)


def avl_dynamics_fd(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
    solver_parameters: dict[str, Any] = {},
) -> None:
    if state.trim == {}:
        print("Trimming the plane")
        aoa_min = -10
        aoa_max = 15
        num_aoa = (aoa_max - aoa_min) + 1
        angles = np.linspace(aoa_min, aoa_max, num_aoa)

        from ICARUS.solvers.AVL import avl_polars

        avl_polars(
            plane=plane,
            state=state,
            solver2D=solver2D,
            angles=angles,
            solver_parameters=solver_parameters,
        )
    if state.epsilons == {}:
        print("Calculating the epsilons")
        state.add_all_pertrubations("Central")
        print(state.epsilons)
    finite_difs(plane=plane, state=state, solver2D=solver2D, solver_parameters=solver_parameters)
    process_avl_dynamics_fd(plane, state)


def process_avl_dynamics_fd(plane: Airplane, state: State) -> DataFrame:
    """Process the pertrubation results from the AVL solver

    Args:
        plane (Airplane): Airplane Object
        state (State): Plane State to load results to

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation

    """
    from ICARUS.solvers.AVL import finite_difs_post

    forces: DataFrame = finite_difs_post(plane, state)

    state.set_pertrubation_results(forces, "AVL")
    state.stability_fd()
    # Save the state
    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )
    state.save(CASEDIR)
    if plane.name not in DB.vehicles_db.states:
        DB.vehicles_db.states[plane.name] = {}
    DB.vehicles_db.states[plane.name][state.name] = state
    return forces


def process_avl_dynamics_implicit(
    plane: Airplane,
    state: State,
) -> tuple[list[complex], list[complex]]:
    from ICARUS.solvers.AVL import implicit_dynamics_post

    impl_long, impl_late = implicit_dynamics_post(plane, state)
    return impl_long, impl_late
