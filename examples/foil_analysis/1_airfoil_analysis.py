import os
import time

import numpy as np
from matplotlib import pyplot as plt

from ICARUS import INSTALL_DIR
from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database
from ICARUS.solvers.OpenFoam.files.setup_case import MeshType
from ICARUS.solvers.Xfoil import XfoilSolverParameters


def main() -> None:
    """Main function to run multiple airfoil simulations"""
    start_time: float = time.time()

    # SETUP DB CONNECTION
    # CHANGE THIS TO YOUR DATABASE FOLDER
    database_folder = os.path.join(INSTALL_DIR, "Data")

    database_folder = os.path.join(INSTALL_DIR, "Data")
    DB = Database(database_folder)

    DB.load_all_data()

    # RUN SETUP
    calcXFoil: bool = True
    calcF2W: bool = True
    calcOpenFoam: bool = True

    print("Running:")
    print(f"\tFoil2Wake section: {calcF2W}")
    print(f"\tXfoil: {calcXFoil}")
    print(f"\tOpenfoam: {calcOpenFoam}")

    print(f"Total number of loaded airfoils {len(list(DB.airfoils.keys()))}")
    print(
        f"Total number of computed airfoil polars {len(list(DB.airfoil_polars.keys()))}",
    )

    # airfoil_names: list[str] = ["0015", "0008", "0012", "2412", "4415"]
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
    reynolds: FloatArray = np.linspace(
        start=reynolds_min,
        stop=reynolds_max,
        num=12,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -8
    aoa_max: float = 14
    num_of_angles: int = int((aoa_max - aoa_min) * 2 + 1)
    angles: FloatArray = np.linspace(
        start=aoa_min,
        stop=aoa_max,
        num=num_of_angles,
    )

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

        # Foil2Wake
        if calcF2W:
            f2w_stime: float = time.time()
            from ICARUS.solvers.Foil2Wake import Foil2Wake

            f2w_s: Foil2Wake = Foil2Wake()

            fw_analysis = f2w_s.aseq

            # Set Inputs
            f2w_inputs = fw_analysis.get_analysis_input()
            f2w_inputs.airfoil = airfoil
            f2w_inputs.reynolds = reynolds
            f2w_inputs.mach = MACH
            f2w_inputs.angles = angles

            f2w_solver_parameters = f2w_s.get_solver_parameters()
            f2w_solver_parameters.f_trip_upper = 0.1
            f2w_solver_parameters.f_trip_low = 1.0
            f2w_solver_parameters.Ncrit = Ncrit
            f2w_solver_parameters.iterations = 250
            f2w_solver_parameters.timestep = 0.1

            _ = f2w_s.execute(
                analysis=fw_analysis,
                inputs=f2w_inputs,
                solver_parameters=f2w_solver_parameters,
            )

            f2w_etime: float = time.time()

            print(f"Foil2Wake completed in {f2w_etime - f2w_stime} seconds")

        # XFoil
        if calcXFoil:
            xfoil_stime: float = time.time()
            from ICARUS.solvers.Xfoil import Xfoil

            xfoil = Xfoil()

            # Import Analysis
            # 0) Sequential Angle run for multiple reynolds with zeroing of the boundary layer between angles,
            # 1) Sequential Angle run for multiple reynolds
            xf_analysis = xfoil.aseq  # Run

            # Get Options
            xfoil_inputs = xf_analysis.get_analysis_input()

            # Set Options
            xfoil_inputs.airfoil = airfoil
            xfoil_inputs.reynolds = reynolds
            xfoil_inputs.mach = MACH
            xfoil_inputs.max_aoa = aoa_max
            xfoil_inputs.min_aoa = aoa_min
            xfoil_inputs.aoa_step = 0.5
            # xfoil_options.angles = angles  # For options 2 and 3

            # Set Solver Options
            xfoil_solver_parameters: XfoilSolverParameters = (
                xfoil.get_solver_parameters()
            )
            xfoil_solver_parameters.max_iter = 500

            xfoil_solver_parameters.Ncrit = Ncrit
            xfoil_solver_parameters.xtr = (0.1, 0.2)
            xfoil_solver_parameters.print = False
            # xfoil.print_solver_parameters()

            # RUN and SAVE
            xfoil.execute(
                analysis=xf_analysis,
                inputs=xfoil_inputs,
                solver_parameters=xfoil_solver_parameters,
            )

            xfoil_etime: float = time.time()
            print(f"XFoil completed in {xfoil_etime - xfoil_stime} seconds")

        # OpenFoam
        if calcOpenFoam:
            of_stime: float = time.time()
            for reyn in reynolds:
                print(f"Running OpenFoam for Re={reyn}")
                from ICARUS.solvers.OpenFoam.open_foam import OpenFoam

                open_foam = OpenFoam()

                # Import Analysis
                of_analysis = open_foam.get_analyses()[0]  # Run

                # Get Options
                of_inputs = of_analysis.get_analysis_input()

                # Set Options
                of_inputs.airfoil = airfoil
                of_inputs.angles = angles
                of_inputs.reynolds = reyn
                of_inputs.mach = MACH

                # Set Solver Options
                of_solver_parameters = open_foam.get_solver_parameters()
                of_solver_parameters.mesh_type = MeshType.structAirfoilMesher
                of_solver_parameters.max_iterations = 100
                of_solver_parameters.silent = False
                # xfoil.print_solver_parameters()

                # RUN
                open_foam.execute(
                    analysis=of_analysis,
                    inputs=of_inputs,
                    solver_parameters=of_solver_parameters,
                )
            of_etime: float = time.time()
            print(f"OpenFoam completed in {of_etime - of_stime} seconds")

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
            polar.save_polar_plot_img(airfoil_folder)

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
