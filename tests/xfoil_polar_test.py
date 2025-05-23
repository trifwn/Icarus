import os
import time

import matplotlib.pyplot as plt
import numpy as np

from ICARUS.airfoils import Airfoil
from ICARUS.computation.solvers import Solver
from ICARUS.core.base_types import Struct
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database


def xfoil_run() -> None:
    """Main function to run multiple airfoil simulations"""
    # SETUP DB CONNECTION
    # CHANGE THIS TO YOUR DATABASE FOLDER
    database_folder = ".\\Data"

    # Load the database
    DB = Database(database_folder)
    DB.load_all_data()

    airfoils_to_compute: list[Airfoil] = []
    airfoils_to_compute.append(DB.get_airfoil("NACA4415"))
    airfoils_to_compute.append(DB.get_airfoil("NACA4412"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0008"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0012"))
    airfoils_to_compute.append(DB.get_airfoil("NACA0015"))
    airfoils_to_compute.append(DB.get_airfoil("NACA2412"))
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
    reynolds = np.linspace(
        start=reynolds_min,
        stop=reynolds_max,
        num=12,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -8
    aoa_max: float = 14
    # Transition to turbulent Boundary Layer
    # ftrip_up: dict[str, float] = {"pos": 0.1, "neg": 1.0}
    # ftrip_low: dict[str, float] = {"pos": 0.1, "neg": 1.0}
    Ncrit = 9
    ###########################################
    for airfoil in airfoils_to_compute:
        # Get airfoil
        airfoil.repanel_spl(200)
        print(f"\nRunning airfoil {airfoil.name}\n")
        xfoil_stime: float = time.time()
        from ICARUS.computation.solvers.Xfoil.xfoil import Xfoil

        xfoil: Solver = Xfoil()

        # Import Analysis
        # 0) Sequential Angle run for multiple reynolds with zeroing of the boundary layer between angles,
        # 1) Sequential Angle run for multiple reynolds
        analysis = xfoil.get_analyses_names()[1]  # Run
        xfoil.select_analysis(analysis)

        # Get Options
        xfoil_options: Struct = xfoil.get_analysis_options()
        xfoil_solver_parameters: Struct = xfoil.get_solver_parameters()

        # Set Options
        xfoil_options.airfoil = airfoil
        xfoil_options.reynolds = reynolds
        xfoil_options.mach = MACH
        xfoil_options.max_aoa = aoa_max
        xfoil_options.min_aoa = aoa_min
        xfoil_options.aoa_step = 0.5
        # xfoil_options.angles = angles  # For options 2 and 3

        # Set Solver Options
        xfoil_solver_parameters.max_iter = 500

        xfoil_solver_parameters.Ncrit = Ncrit
        xfoil_solver_parameters.xtr = (0.2, 0.2)
        xfoil_solver_parameters.print = False
        # xfoil.print_solver_options()

        # RUN and SAVE
        xfoil.define_analysis(xfoil_options, xfoil_solver_parameters)
        xfoil.print_analysis_options()
        xfoil.execute(parallel=True)

        xfoil_etime: float = time.time()
        print(f"Airfoil {airfoil.name} completed in {xfoil_etime - xfoil_stime} seconds")

        try:
            # Get polar
            polar = DB.get_airfoil_polars(airfoil)
            airfoil_folder = os.path.join(DB.DB2D, "images")
            os.makedirs(airfoil_folder, exist_ok=True)
            polar.plot()
            polar.save_polar_plot_img(airfoil_folder, "xfoil")
            plt.show(block=False)

        except Exception as e:
            print(f"Error saving polar plot. Got: {e}")
