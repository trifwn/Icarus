"""Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""

import os
import time

import numpy as np
from pandas import DataFrame

from ICARUS.computation.solvers import Solver
from ICARUS.computation.solvers.XFLR5.polars import read_polars_2d
from ICARUS.core.base_types import Struct
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane

# DB CONNECTION
database_folder = os.path.join("/mnt/e/ICARUS", "Data")
DB = Database(database_folder)
read_polars_2d(os.path.join(DB.EXTERNAL_DB, "2D"))


def main(GNVP_VERSION: int) -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # Get Plane
    planes: list[Airplane] = []

    # from vehicles.Planes.e190_takeoff import e190_takeoff_generator
    # embraer_to: Airplane = e190_takeoff_generator(name="e190_to_3")
    # planes.append(embraer_to)

    # from vehicles.Planes.e190_cruise import e190_cruise

    # embraer_cr: Airplane = e190_cruise(name="e190_cr_3")
    # planes.append(embraer_cr)
    # planes.append(embraer_to)

    # filename: str = "Data/3d_Party/plane_1.xml"
    # airplane = parse_xfl_project(filename)
    # from Planes.e190_cruise import e190_cruise
    from Planes.hermes import hermes

    name = "hermes"
    hermes_3: Airplane = hermes(name=name)
    planes.append(hermes_3)

    timestep: dict[str, float] = {name: 1e-3}
    maxiter: dict[str, int] = {name: 100}
    UINF: dict[str, float] = {name: 20}
    ALTITUDE: dict[str, int] = {name: 0}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, int] = {name: 273 + 15}

    STATIC_ANALYSIS: dict[str, float] = {name: True}
    DYNAMIC_ANALYSIS: dict[str, float] = {name: False}

    if GNVP_VERSION == 7:
        from ICARUS.computation.solvers.GenuVP import GenuVP7

        gnvp: Solver = GenuVP7()
    elif GNVP_VERSION == 3:
        from ICARUS.computation.solvers.GenuVP import GenuVP3

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

            analysis: str = gnvp.get_analyses_names()[0]
            gnvp.select_analysis(analysis)
            options: Struct = gnvp.get_analysis_options()
            solver_parameters: Struct = gnvp.get_solver_parameters()
            AOA_MIN = -6
            AOA_MAX = 10
            NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
            angles: FloatArray = np.linspace(
                AOA_MIN,
                AOA_MAX,
                NO_AOA,
            )

            options.plane = plane
            options.solver2D = "XFLR"
            options.state = state
            options.maxiter = maxiter[plane.name]
            options.timestep = timestep[plane.name]
            options.angles = angles

            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False

            gnvp.define_analysis(options, solver_parameters)
            gnvp.print_analysis_options()

            polars_time: float = time.time()
            gnvp.execute(parallel=True)
            print(
                f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
            )
            plane.save()

            from ICARUS.computation.solvers.GenuVP import process_gnvp_polars

            process_gnvp_polars(plane, state, GNVP_VERSION)

            # from ICARUS.computation.solvers.AVL import avl_polars

            # avl_polars(plane, state, "XFLR", angles)

            # from ICARUS.visualization.airplane import plot_airplane_polars

            # solvers = [
            #     "GenuVP3 Potential" if GNVP_VERSION == 3 else "GenuVP7 Potential",
            #     "GenuVP3 2D" if GNVP_VERSION == 3 else "GenuVP7 2D",
            #     "GenuVP3 ONERA" if GNVP_VERSION == 3 else "GenuVP7 ONERA",
            #     'AVL',
            # ]
            # axs, fig = plot_airplane_polars(
            #     [airplane.name],
            #     solvers,
            #     plots=[["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"]],
            #     size=(6, 7),
            # )
            # Pause to see the plots for 2 seconds
            time.sleep(2)

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
                print("Got errro")
                raise (error)
                continue

            # ### Pertrubations
            epsilons = None

            unstick.add_all_pertrubations("Central", epsilons)
            unstick.get_pertrub()

            # Define Analysis for Pertrubations
            # 3 Pertrubations Serial
            # 4 Pertrubations Parallel
            # 5 Sesitivity Analysis Serial
            # 6 Sesitivity Analysis Parallel
            analysis = gnvp.get_analyses_names()[1]  # Pertrubations PARALLEL
            print(f"Selecting Analysis: {analysis}")
            gnvp.select_analysis(analysis)

            options = gnvp.get_analysis_options(verbose=False)
            solver_parameters = gnvp.get_solver_parameters(verbose=False)

            if options is None:
                raise ValueError("Options not set")
            # Set Options
            options.plane = plane
            options.state = unstick
            options.solver2D = "Xfoil"
            options.maxiter = maxiter[plane.name]
            options.timestep = timestep[plane.name]
            # options.angle = unstick.trim["AoA"]

            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False
            # Run Analysis
            gnvp.define_analysis(options, solver_parameters)
            gnvp.print_analysis_options()

            pert_time: float = time.time()
            print("Running Pertrubations")
            gnvp.execute(parallel=True)
            print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

            # Get Results And Save
            _ = gnvp.get_results()
    # print time program took
    print(f"WORKFLOW FOR {GNVP_VERSION} TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main(3)
    main(7)
