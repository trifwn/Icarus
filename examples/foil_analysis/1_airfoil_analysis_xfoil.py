import os

import numpy as np
from matplotlib import pyplot as plt

from ICARUS.airfoils import Airfoil
from ICARUS.computation.core.types import ExecutionMode
from ICARUS.core.types import FloatArray
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database
from ICARUS.settings import INSTALL_DIR
from ICARUS.solvers.Xfoil.xfoil import XfoilSolverParameters


def main() -> None:
    """Main function to run multiple airfoil simulations"""

    # SETUP DB CONNECTION
    # CHANGE THIS TO YOUR DATABASE FOLDER
    database_folder = os.path.join(
        INSTALL_DIR,
        "Data",
    )

    # Load the database
    DB = Database(database_folder)
    DB.load_all_data()

    # RUN SETUP
    airfoil_names: list[str] = ["0015", "0008", "0012", "2412", "4415"]
    airfoils_to_compute: list[Airfoil] = []
    for airfoil_name in airfoil_names:
        airfoil = DB.get_airfoil(f"NACA{airfoil_name}")
        airfoil.repanel_spl(160)
        airfoils_to_compute.append(airfoil)

    print(f"Computing: {len(airfoils_to_compute)}")

    # airfoils_to_compute = airfoils_to_compute[0]
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

    from ICARUS.solvers.Xfoil.xfoil import Xfoil

    xfoil = Xfoil()
    print(xfoil)

    # Import Analysis
    # 0) Sequential Angle run for multiple reynolds with zeroing of the boundary layer between angles,
    # 1) Sequential Angle run for multiple reynolds
    analysis = xfoil.aseq

    # Set Options
    xfoil_inputs = analysis.get_analysis_input()
    xfoil_inputs.airfoil = airfoils_to_compute
    xfoil_inputs.reynolds = reynolds
    xfoil_inputs.mach = MACH
    xfoil_inputs.max_aoa = aoa_max
    xfoil_inputs.min_aoa = aoa_min
    xfoil_inputs.aoa_step = aoa_step

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
        execution_mode=ExecutionMode.THREADING,
    )

    try:
        for airfoil in airfoils_to_compute:
            # Get polar
            polar = DB.get_airfoil_polars(airfoil)
            airfoil_folder, _, _ = DB.generate_airfoil_directories(airfoil)
            polar.plot()
            plt.show(block=True)
            polar.save_polar_plot_img(airfoil_folder)

    except Exception as e:
        print(f"Error saving polar plot. Got: {e}")
        raise (e)


if __name__ == "__main__":
    main()
