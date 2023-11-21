"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import time

import numpy as np
from pandas import DataFrame

from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Solvers.Airplane.gnvp3 import get_gnvp3
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Workers.solver import Solver


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # DB CONNECTION
    from ICARUS.Input_Output.XFLR5.polars import read_polars_2d
    from ICARUS.Database import EXTERNAL_DB

    read_polars_2d(EXTERNAL_DB)

    # # Get Plane
    planes: list[Airplane] = []

    # from Vehicles.Planes.e190_takeoff import e190_takeoff_generator
    # embraer_to: Airplane = e190_takeoff_generator(name="e190_to_3")
    # planes.append(embraer_to)

    # from Vehicles.Planes.e190_cruise import e190_cruise

    # embraer_cr: Airplane = e190_cruise(name="e190_cr_3")
    # planes.append(embraer_cr)
    # planes.append(embraer_to)

    from ICARUS.Input_Output.XFLR5.parser import parse_xfl_project

    filename: str = "Data/3d_Party/plane_1.xml"
    airplane = parse_xfl_project(filename)
    # from Vehicles.Planes.hermes import hermes
    # hermes_3: Airplane = hermes(name="hermes_3_2")

    planes.append(airplane)
    # embraer.visualize()

    timestep: dict[str, float] = {"e190_to_3": 10, "e190_cr_3": 10, "plane_1": 1e-3}
    maxiter: dict[str, int] = {"e190_to_3": 50, "e190_cr_3": 50, "plane_1": 100}
    UINF: dict[str, float] = {"e190_to_3": 20, "e190_cr_3": 232, "plane_1": 20}
    ALTITUDE: dict[str, int] = {"e190_cr_3": 12000, "e190_to_3": 0, "plane_1": 0}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, int] = {"e190_cr_3": 273 - 50, "e190_to_3": 273 + 15, "plane_1": 273 + 15}

    STATIC_ANALYSIS: dict[str, float] = {"e190_to_3": False, "e190_cr_3": False, "plane_1": True}
    DYNAMIC_ANALYSIS: dict[str, float] = {"e190_to_3": False, "e190_cr_3": False, "plane_1": True}

    # Get Solver
    GNVP_VERSION = 3
    if GNVP_VERSION == 7:
        from ICARUS.Solvers.Airplane.gnvp7 import get_gnvp7

        gnvp: Solver = get_gnvp7()
    elif GNVP_VERSION == 3:
        from ICARUS.Solvers.Airplane.gnvp3 import get_gnvp3

        gnvp = get_gnvp3()
    else:
        raise ValueError("GNVP VERSION NOT FOUND")

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(ALTITUDE[airplane.name], TEMPERATURE[airplane.name])
        print(EARTH_ISA)

        if STATIC_ANALYSIS[airplane.name]:
            # ## AoA Run
            # 0: Single Angle of Attack (AoA) Run
            # 1: Angles Sequential
            # 2: Angles Parallel

            analysis: str = gnvp.available_analyses_names()[2]
            gnvp.set_analyses(analysis)
            options: Struct = gnvp.get_analysis_options(verbose=False)
            solver_parameters: Struct = gnvp.get_solver_parameters()

            AOA_MIN = -5
            AOA_MAX = 4
            NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
            angles: FloatArray = np.linspace(
                AOA_MIN,
                AOA_MAX,
                NO_AOA,
            )

            airplane.define_dynamic_pressure(UINF[airplane.name], EARTH_ISA.air_density)

            options.plane.value = airplane
            options.environment.value = EARTH_ISA
            options.solver2D.value = "Xfoil"
            options.maxiter.value = maxiter[airplane.name]
            options.timestep.value = timestep[airplane.name]
            options.u_freestream.value = UINF[airplane.name]
            options.angles.value = angles

            solver_parameters.Use_Grid.value = True
            solver_parameters.Split_Symmetric_Bodies.value = False

            gnvp.print_analysis_options()

            polars_time: float = time.time()
            gnvp.run()
            print(
                f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
            )
            polars: DataFrame | int = gnvp.get_results()
            airplane.save()

            from ICARUS.Visualization.airplane.db_polars import plot_airplane_polars

            solvers = [
                "GNVP3 Potential" if GNVP_VERSION == 3 else "GNVP7 Potential",
                "GNVP3 2D" if GNVP_VERSION == 3 else "GNVP7 2D",
                "GNVP3 ONERA" if GNVP_VERSION == 3 else "GNVP7 ONERA",
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
            polars = DB.vehicles_db.raw_data[airplane.name]
            print(polars)
            if not isinstance(polars, DataFrame):
                raise ValueError(f"Polars for {airplane.name} not found in DB")

            try:
                unstick = State("Unstick", airplane, polars, EARTH_ISA, preffered_polar="Potential")
            except Exception as error:
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
            analysis = gnvp.available_analyses_names()[4]  # Pertrubations PARALLEL
            print(f"Selecting Analysis: {analysis}")
            gnvp.set_analyses(analysis)

            options = gnvp.get_analysis_options(verbose=False)
            solver_parameters = gnvp.get_solver_parameters()

            if options is None:
                raise ValueError("Options not set")
            # Set Options
            options.plane.value = airplane
            options.state.value = unstick
            options.environment.value = EARTH_ISA
            options.solver2D.value = "Xfoil"
            options.maxiter.value = maxiter[airplane.name]
            options.timestep.value = timestep[airplane.name]
            options.u_freestream.value = unstick.trim["U"]
            options.angle.value = unstick.trim["AoA"]

            solver_parameters.Use_Grid.value = True
            solver_parameters.Split_Symmetric_Bodies.value = False
            # Run Analysis
            gnvp.print_analysis_options()

            pert_time: float = time.time()
            print("Running Pertrubations")
            gnvp.run()
            print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

            # Get Results And Save
            _ = gnvp.get_results()
            unstick.save()

            # Sensitivity ANALYSIS
            # ADD SENSITIVITY ANALYSIS

    # print time program took
    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
