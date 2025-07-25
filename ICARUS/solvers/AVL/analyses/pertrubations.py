import numpy as np
from pandas import DataFrame

from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL import AVLParameters
from ICARUS.vehicle import Airplane

from ..files.cases import AVLRunSetup
from ..files.cases import avl_run_cases
from ..files.dynamics import implicit_eigs
from ..files.input import make_input_files


def avl_stability_implicit(
    plane: Airplane,
    state: State,
    solver_parameters: AVLParameters = AVLParameters(),
) -> None:
    implicit_eigs(
        plane=plane,
        state=state,
        solver_parameters=solver_parameters,
    )


def avl_stability_fd(
    plane: Airplane,
    state: State,
    solver_parameters: AVLParameters = AVLParameters(),
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
            angles=angles,
            solver_parameters=solver_parameters,
        )

    if state.epsilons == {}:
        print("Calculating the epsilons")
        state.add_all_pertrubations("Central")
        print(state.epsilons)

    DB = Database.get_instance()
    case_directory = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )

    make_input_files(
        directory=case_directory,
        plane=plane,
        state=state,
        solver_parameters=solver_parameters,
    )
    run_setup = AVLRunSetup.stability_fd(
        name=f"{plane.name}_fd.run",
        state=state,
        plane=plane,
    )
    avl_run_cases(case_directory, plane, run_setup)

    process_avl_dynamics_fd(plane, state, run_setup)


def process_avl_dynamics_fd(
    plane: Airplane,
    state: State,
    run_setup: AVLRunSetup,
) -> DataFrame:
    """Process the pertrubation results from the AVL solver

    Args:
        plane (Airplane): Airplane Object
        state (State): Plane State to load results to

    Returns:
        DataFrame: DataFrame with the forces for each pertrubation simulation

    """
    from ICARUS.solvers.AVL.post_process import finite_difs_post

    forces: DataFrame = finite_difs_post(plane, state, run_setup)

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
    from ICARUS.solvers.AVL.post_process import implicit_dynamics_post

    DB = Database.get_instance()
    CASEDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
    )
    # Save the state
    plane.save()
    state.save(CASEDIR)
    if plane.name not in DB.vehicles_db.states:
        DB.vehicles_db.states[plane.name] = {}
    DB.vehicles_db.states[plane.name][state.name] = state

    eigen_modes = implicit_dynamics_post(plane, state)
    # return impl_long, impl_late
    long = [e for e in eigen_modes if e.matrix_values["u"] == 0]
    late = [e for e in eigen_modes if e.matrix_values["u"] != 0]
    longitudal_eigenvals = [mode.eigenvalue for mode in long]
    lateral_eigenvals = [mode.eigenvalue for mode in late]
    return longitudal_eigenvals, lateral_eigenvals
