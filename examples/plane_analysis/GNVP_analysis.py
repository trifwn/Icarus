"""Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""

import copy
import os
import time

import numpy as np
from pandas import DataFrame

from ICARUS import INSTALL_DIR
from ICARUS.computation.core import ExecutionMode
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.solvers.XFLR5.read_xflr5_polars import read_XFLR5_airfoil_polars
from ICARUS.vehicle import Airplane

# DB CONNECTION
database_folder = os.path.join(INSTALL_DIR, "Data")
DB = Database(database_folder)
read_XFLR5_airfoil_polars(os.path.join(DB.EXTERNAL_DB, "2D"))


def main(GNVP_VERSION: int) -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # Get Plane
    planes: list[Airplane] = []

    from Planes.hermes import hermes

    name = "hermes_low_fidelity"
    hermes_3: Airplane = hermes(name=name)
    planes.append(hermes_3)

    UINF: dict[str, float] = {name: 20}
    ALTITUDE: dict[str, int] = {name: 0}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, int] = {name: 273 + 15}

    STATIC_ANALYSIS: dict[str, float] = {name: True}
    DYNAMIC_ANALYSIS: dict[str, float] = {name: True}

    if GNVP_VERSION == 7:
        from ICARUS.solvers.GenuVP import GenuVP7

        gnvp = GenuVP7()
    elif GNVP_VERSION == 3:
        from ICARUS.solvers.GenuVP import GenuVP3

        gnvp = GenuVP3()
    else:
        raise ValueError("GNVP VERSION NOT FOUND")

    for plane in planes:
        print("--------------------------------------------------")
        print(f"Running {plane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(
            ALTITUDE[plane.name],
            TEMPERATURE[plane.name],
        )
        state = State(
            name="Unstick",
            airplane=plane,
            environment=EARTH_ISA,
            u_freestream=UINF[plane.name],
        )
        print(EARTH_ISA)
        state.save(os.path.join(DB.DB3D, plane.directory))

        if STATIC_ANALYSIS[plane.name]:
            # ## AoA Run
            # 0: Single Angle of Attack (AoA) Run
            # 1: Angles Sequential
            # 2: Angles Parallel

            polar_analysis = gnvp.aseq
            inputs = polar_analysis.get_analysis_input()
            AOA_MIN = -6
            AOA_MAX = 10
            NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
            angles: FloatArray = np.linspace(
                AOA_MIN,
                AOA_MAX,
                NO_AOA,
            )

            inputs.plane = plane
            inputs.state = state
            inputs.angles = angles

            solver_parameters = gnvp.get_solver_parameters()
            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False
            solver_parameters.timestep = 0.1
            solver_parameters.iterations = 40

            polars_time: float = time.time()
            gnvp.execute(
                analysis=polar_analysis,
                inputs=inputs,
                solver_parameters=solver_parameters,
                execution_mode=ExecutionMode.MULTIPROCESSING,
            )
            print(
                f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
            )
            plane.save()

            from ICARUS.solvers.GenuVP import process_gnvp_polars

            process_gnvp_polars(plane, state, GNVP_VERSION)

        if DYNAMIC_ANALYSIS[plane.name]:
            # # Dynamics
            # ### Define and Trim Plane
            forces = DB.get_vehicle_polars(plane)
            if not isinstance(forces, DataFrame):
                raise ValueError(f"Polars for {plane.name} not found in DB")
            try:
                state.add_polar(
                    polar=forces,
                    polar_prefix=f"GenuVP{GNVP_VERSION} Potential",
                    is_dimensional=False,
                )
                unstick = state
            except Exception as error:
                raise (error)
                continue

            # ### Pertrubations
            epsilons = None

            unstick.add_all_pertrubations("Central", epsilons)
            unstick.print_pertrubations()

            # Define Analysis for Pertrubations
            # 3 Pertrubations Serial
            # 4 Pertrubations Parallel
            # 5 Sesitivity Analysis Serial
            # 6 Sesitivity Analysis Parallel
            stability_analysis = gnvp.stability  # Pertrubations PARALLEL
            print("Selecting Analysis:")
            print(stability_analysis)

            stability_inputs = stability_analysis.get_analysis_input(verbose=False)
            stability_inputs.plane = plane
            stability_inputs.state = unstick
            stability_inputs.disturbances = copy.copy(unstick.disturbances)

            solver_parameters = gnvp.get_solver_parameters(verbose=False)
            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False
            solver_parameters.timestep = 0.1
            solver_parameters.iterations = 40

            # Run Analysis
            pert_time: float = time.time()
            print("Running Pertrubations")
            _ = gnvp.execute(
                analysis=stability_analysis,
                inputs=stability_inputs,
                solver_parameters=solver_parameters,
                execution_mode=ExecutionMode.MULTIPROCESSING,
                # progress_monitor= None
            )
            print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

    # print time program took
    print(f"WORKFLOW FOR GenuVP{GNVP_VERSION} TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main(3)
    # main(7)
