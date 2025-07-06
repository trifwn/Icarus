import os

import numpy as np
from matplotlib import pyplot as plt

from ICARUS.airfoils import Airfoil
from ICARUS.computation.core.types import ExecutionMode
from ICARUS.core.types import FloatArray
from ICARUS.core.units import calc_reynolds
from ICARUS.database import Database
from ICARUS.settings import INSTALL_DIR
from ICARUS.solvers.Foil2Wake.f2w_section import Foil2WakeSolverParameters


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
    airfoil_names: list[str] = ["0015"]  # "0008", "0012", "2412", "4415"]
    airfoils_to_compute: list[Airfoil] = []
    for airfoil_name in airfoil_names:
        airfoil = DB.get_airfoil(f"NACA{airfoil_name}")
        airfoil.repanel_spl(200)
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
        num=3,
    )

    # Transition to turbulent Boundary Layer
    Ncrit = 9
    angles = np.linspace(-8, 12, 21)  # Angles of attack in degrees

    from ICARUS.solvers.Foil2Wake import Foil2Wake

    f2w = Foil2Wake()
    print(f2w)

    # Import Analysis
    # 0) Sequential Angle run for multiple reynolds with zeroing of the boundary layer between angles,
    # 1) Sequential Angle run for multiple reynolds
    analysis = f2w.aseq

    # Set Options
    f2w_inputs = analysis.get_analysis_input()
    f2w_inputs.airfoil = airfoils_to_compute
    f2w_inputs.reynolds = reynolds
    f2w_inputs.mach = MACH
    f2w_inputs.angles = angles

    # Set Solver Options
    f2w_solver_parameters: Foil2WakeSolverParameters = f2w.get_solver_parameters()
    f2w_solver_parameters.iterations = 350
    f2w.solver_parameters.boundary_layer_iteration_start = 349
    f2w_solver_parameters.Ncrit = Ncrit
    f2w_solver_parameters.f_trip_upper = 0.1
    f2w_solver_parameters.f_trip_low = 0.2

    # RUN and SAVE
    f2w.execute(
        analysis=analysis,
        inputs=f2w_inputs,
        solver_parameters=f2w_solver_parameters,
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
