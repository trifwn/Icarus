"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import time

import numpy as np
from pandas import DataFrame

from ICARUS.computation.solvers.solver import Solver
from ICARUS.core.struct import Struct
from ICARUS.core.types import FloatArray
from ICARUS.database import DB
from ICARUS.environment.definition import EARTH_ISA
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # DB CONNECTION
    from ICARUS.computation.solvers.XFLR5.polars import read_polars_2d
    from ICARUS.database import EXTERNAL_DB

    read_polars_2d(EXTERNAL_DB)

    # # Get Plane
    planes: list[Airplane] = []

    # from vehicles.Planes.e190_takeoff import e190_takeoff_generator
    # embraer_to: Airplane = e190_takeoff_generator(name="e190_to_3")
    # planes.append(embraer_to)

    # from vehicles.Planes.e190_cruise import e190_cruise

    # embraer_cr: Airplane = e190_cruise(name="e190_cr_3")
    # planes.append(embraer_cr)
    # planes.append(embraer_to)

    from ICARUS.computation.solvers.XFLR5.parser import parse_xfl_project

    # filename: str = "Data/3d_Party/plane_1.xml"
    # airplane = parse_xfl_project(filename)
    from vehicles.planes.e190_cruise import e190_cruise

    hermes_3: Airplane = e190_cruise(name="E190")
    planes.append(hermes_3)

    # planes.append(airplane)
    # embraer.visualize()

    timestep: dict[str, float] = {"E190": 1e-3}
    maxiter: dict[str, int] = {"E190": 100}
    UINF: dict[str, float] = {"E190": 20}
    ALTITUDE: dict[str, int] = {"E190": 0}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, int] = {"E190": 273 + 15}

    STATIC_ANALYSIS: dict[str, float] = {"E190": True}
    DYNAMIC_ANALYSIS: dict[str, float] = {"E190": False}

    # Get Solver
    GNVP_VERSION = 3
    if GNVP_VERSION == 7:
        from ICARUS.computation.solvers.GenuVP.gnvp7 import GenuVP7

        gnvp: Solver = GenuVP7()
    elif GNVP_VERSION == 3:
        from ICARUS.computation.solvers.GenuVP.gnvp3 import GenuVP3

        gnvp = GenuVP3()
    else:
        raise ValueError("GNVP VERSION NOT FOUND")

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(ALTITUDE[airplane.name], TEMPERATURE[airplane.name])
        state = State(
            name="Unstick",
            airplane=airplane,
            environment=EARTH_ISA,
            u_freestream=UINF[airplane.name],
        )
        print(EARTH_ISA)

        if STATIC_ANALYSIS[airplane.name]:
            # ## AoA Run
            # 0: Single Angle of Attack (AoA) Run
            # 1: Angles Sequential
            # 2: Angles Parallel

            analysis: str = gnvp.get_analyses_names()[0]
            gnvp.select_analysis(analysis)
            options: Struct = gnvp.get_analysis_options()
            solver_parameters: Struct = gnvp.get_solver_parameters()
            AOA_MIN = -5
            AOA_MAX = 4
            NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
            angles: FloatArray = np.linspace(
                AOA_MIN,
                AOA_MAX,
                NO_AOA,
            )

            options.plane = airplane
            options.solver2D = "XFLR"
            options.state = state
            options.maxiter = maxiter[airplane.name]
            options.timestep = timestep[airplane.name]
            options.angles = angles

            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False

            gnvp.define_analysis(options, solver_parameters)
            gnvp.print_analysis_options()

            polars_time: float = time.time()
            gnvp.execute()
            print(
                f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
            )
            polars: DataFrame | int = gnvp.get_results()
            airplane.save()

            from ICARUS.visualization.airplane.db_polars import plot_airplane_polars

            solvers = [
                "GenuVP3 Potential" if GNVP_VERSION == 3 else "GenuVP7 Potential",
                "GenuVP3 2D" if GNVP_VERSION == 3 else "GenuVP7 2D",
                "GenuVP3 ONERA" if GNVP_VERSION == 3 else "GenuVP7 ONERA",
            ]
            axs, fig = plot_airplane_polars(
                [airplane.name],
                solvers,
                plots=[["AoA", "CL"], ["AoA", "CD"], ["AoA", "Cm"]],
                size=(6, 7),
            )
            # Pause to see the plots for 2 seconds
            time.sleep(2)

        if DYNAMIC_ANALYSIS[airplane.name]:
            # # Dynamics
            # ### Define and Trim Plane
            forces = DB.vehicles_db.forces[airplane.name]
            if not isinstance(forces, DataFrame):
                raise ValueError(f"Polars for {airplane.name} not found in DB")
            try:
                state.add_polar(polar=forces, polar_prefix="GenuVP3 2D", is_dimensional=True)
                unstick = state
            except Exception as error:
                print("Got errro")
                print(error)
                continue

            # ### Pertrubations
            # epsilons = {
            #     "u": 0.01,
            #     "w": 0.01,
            #     "q": 0.001,
            #     "theta": 0.01 ,
            #     "v": 0.01,
            #     "p": 0.001,
            #     "r": 0.001,
            #     "phi": 0.001
            # }
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
            options.plane = airplane
            options.state = unstick
            options.solver2D = "Xfoil"
            options.maxiter = maxiter[airplane.name]
            options.timestep = timestep[airplane.name]
            options.angle = unstick.trim["AoA"]

            solver_parameters.Use_Grid = True
            solver_parameters.Split_Symmetric_Bodies = False
            # Run Analysis
            gnvp.define_analysis(options, solver_parameters)
            gnvp.print_analysis_options()

            pert_time: float = time.time()
            print("Running Pertrubations")
            gnvp.execute()
            print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

            # Get Results And Save
            _ = gnvp.get_results()

            # Sensitivity ANALYSIS
            # ADD SENSITIVITY ANALYSIS

    # print time program took
    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
