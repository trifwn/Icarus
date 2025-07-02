import os
import time

import numpy as np
from matplotlib import pyplot as plt

from ICARUS.airfoils import Airfoil
from ICARUS.computation import Solver
from ICARUS.computation.core.types import ExecutionMode
from ICARUS.core.types import FloatArray
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database
from ICARUS.solvers.Xfoil.xfoil import XfoilAseq, XfoilAseqInput, XfoilSolverParameters


def main() -> None:
    """Main function to run multiple airfoil simulations"""
    start_time: float = time.time()

    # SETUP DB CONNECTION
    # CHANGE THIS TO YOUR DATABASE FOLDER
    database_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Data",
    )

    # Load the database
    DB = Database(database_folder)
    DB.load_all_data()

    # RUN SETUP
    calcXFoil: bool = True

    print("Running:")
    print(f"\tXfoil: {calcXFoil}")

    print(f"Total number of loaded airfoils {len(list(DB.airfoils.keys()))}")
    print(
        f"Total number of computed airfoil polars {len(list(DB.airfoil_polars.keys()))}",
    )

    # airfoil_names: list[str] = ["0015", "0008", "0012", "2412", "4415"]
    airfoils_to_compute: list[Airfoil] = []
    airfoils_to_compute.append(DB.get_airfoil("NACA4415"))
    # airfoils_to_compute.append(DB.get_airfoil("NACA4412"))
    # airfoils_to_compute.append(DB.get_airfoil("NACA0008"))
    # airfoils_to_compute.append(DB.get_airfoil("NACA0012"))
    # airfoils_to_compute.append(DB.get_airfoil("NACA0015"))
    # airfoils_to_compute.append(DB.get_airfoil("NACA2412"))
    for airfoil in airfoils_to_compute:
        airfoil.repanel_spl(160)

    print(f"Computing: {len(airfoils_to_compute)}")

    # PARAMETERS FOR ESTIMATION
    chord_max: float = 0.16
    chord_min: float = 0.06
    u_max: float = 35.0
    u_min: float = 5.0
    viscosity: float = 1.56e-5

    # MACH ESTIMATION
    mach_max: float = 0.085
    # mach_min: float = calc_mach(10, speed_of_sound)
    # mach: FloatArray = np.linspace(mach_max, mach_min, 10)
    MACH: float = mach_max

    # REYNOLDS ESTIMATION
    reynolds_max: float = calc_reynolds(u_max, chord_max, viscosity)
    reynolds_min: float = calc_reynolds(u_min, chord_min, viscosity)
    reynolds: FloatArray = np.linspace(
        start=reynolds_min,
        stop=reynolds_max,
        num=12,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -8
    aoa_max: float = 14
    aoa_step: float = 1.0

    # Transition to turbulent Boundary Layer
    # ftrip_up: dict[str, float] = {"pos": 0.1, "neg": 1.0}
    # ftrip_low: dict[str, float] = {"pos": 0.1, "neg": 1.0}
    Ncrit = 9

    #   ############################## START LOOP ###########################################
    for airfoil in airfoils_to_compute:
        # Get airfoil
        airfoil.repanel_spl(200)
        airfoil_stime: float = time.time()
        print(f"\nRunning airfoil {airfoil.name}\n")
        # airfoil.plot()
        # airfoil.repanel(100, distribution="cosine")

        # XFoil
        if calcXFoil:
            xfoil_stime: float = time.time()
            from ICARUS.solvers.Xfoil.xfoil import Xfoil

            xfoil: Solver = Xfoil()
            print(xfoil)

            # Import Analysis
            # 0) Sequential Angle run for multiple reynolds with zeroing of the boundary layer between angles,
            # 1) Sequential Angle run for multiple reynolds
            analysis: XfoilAseq = xfoil.get_analyses()[1]  # Run

            # Set Options
            xfoil_inputs:  XfoilAseqInput = analysis.get_analysis_input()
            xfoil_inputs.airfoil = airfoil
            xfoil_inputs.reynolds = reynolds
            xfoil_inputs.mach = MACH
            xfoil_inputs.max_aoa = aoa_max
            xfoil_inputs.min_aoa = aoa_min
            xfoil_inputs.aoa_step = aoa_step
            # xfoil_options.angles = angles  # For options 2 and 3

            # Set Solver Options
            xfoil_solver_parameters: XfoilSolverParameters = xfoil.get_solver_parameters()
            xfoil_solver_parameters.max_iter = 500
            xfoil_solver_parameters.Ncrit = Ncrit
            xfoil_solver_parameters.xtr = (0.1, 0.2)
            xfoil_solver_parameters.print = False
            # xfoil.print_solver_parameters()

            # RUN and SAVE
            xfoil.execute(
                analysis=analysis,
                inputs=xfoil_inputs,
                solver_parameters=xfoil_solver_parameters,
                execution_mode=ExecutionMode.MULTIPROCESSING,
            )

            xfoil_etime: float = time.time()
            print(f"XFoil completed in {xfoil_etime - xfoil_stime} seconds")

        airfoil_etime: float = time.time()
        print(
            f"Airfoil {airfoil.name} completed in {airfoil_etime - airfoil_stime} seconds",
        )
        try:
            # Get polar
            polar = DB.get_airfoil_polars(airfoil)
            airfoil_folder = os.path.join("Data/images/")
            polar.plot()
            plt.show(block=True)
            polar.save_polar_plot_img(airfoil_folder, "xfoil")

        except Exception as e:
            print(f"Error saving polar plot. Got: {e}")

    ################################ END LOOP ##############################################

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    print("########################################################################")
    print("Program Terminated")
    print("########################################################################")


if __name__ == "__main__":
    main()
