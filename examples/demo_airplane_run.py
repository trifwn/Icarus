"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import time
from re import T

import numpy as np
from demo_aeroplano import airplane_generator
from pandas import DataFrame

from ICARUS.Computation.Solvers.GenuVP.gnvp3 import get_gnvp3
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane

########################################################################################
########################################################################################
###################      PARAMETERS   ##################################################
########################################################################################

name = "aeroplano"
aeroplano: Airplane = airplane_generator(name)

# Angles to run Static Analysis for
AOA_MIN = -5
AOA_MAX = 6
NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
angles: FloatArray = np.linspace(
    AOA_MIN,
    AOA_MAX,
    NO_AOA,
)

# Environment Definition
temperature: float = 273 + 15  # TEMPERATURE IN KELVIN
altitude: float = 0  # ALTITUDE IN METERS
uinf: float = 20  # FREESTREAM VELOCITY IN m/s

# Solver Parameters
timestep: float = 1e-3  # Timestep for the solver
maxiter: int = 300  # Number of time iterations for the solver
airfoil_solver: str = "Foil2Wake"  # Solver to use for the airfoil

# Dynamic Analysis
run_dynamic_analysis: bool = True  # Run the dynamic analysis?

# ### Pertrubation Definition
epsilons: dict[str, float] = {
    "u": 0.01,
    "w": 0.01,
    "q": 0.001,
    "theta": 0.01,
    "v": 0.01,
    "p": 0.001,
    "r": 0.001,
    "phi": 0.001,
}

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
    from ICARUS.Database import EXTERNAL_DB

    read_polars_2d(EXTERNAL_DB)

    ## Get Plane
    planes: list[Airplane] = []
    planes.append(aeroplano)

    TIMESTEP: dict[str, float] = {f"{name}": timestep}
    MAXITER: dict[str, int] = {f"{name}": maxiter}
    UINF: dict[str, float] = {f"{name}": uinf}
    ALTITUDE: dict[str, float] = {f"{name}": altitude}
    TEMPERATURE: dict[str, float] = {f"{name}": temperature}
    DYNAMICS: dict[str, bool] = {f"{name}": run_dynamic_analysis}

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(ALTITUDE[airplane.name], TEMPERATURE[airplane.name])
        print(EARTH_ISA)

        # # Get Solver
        gnvp3: Solver = get_gnvp3()

        # ## AoA Run
        # 0: Single Angle of Attack (AoA) Run
        # 1: Angles Sequential
        # 2: Angles Parallel

        analysis: str = gnvp3.available_analyses_names()[2]
        gnvp3.set_analyses(analysis)
        options: Struct = gnvp3.get_analysis_options(verbose=False)
        solver_parameters: Struct = gnvp3.get_solver_parameters()

        airplane.define_dynamic_pressure(UINF[airplane.name], EARTH_ISA.air_density)

        options.plane.value = airplane
        options.environment.value = EARTH_ISA
        options.solver2D.value = airfoil_solver
        options.maxiter.value = MAXITER[airplane.name]
        options.timestep.value = TIMESTEP[airplane.name]
        options.u_freestream.value = UINF[airplane.name]
        options.angles.value = angles

        solver_parameters.Use_Grid.value = True
        solver_parameters.Split_Symmetric_Bodies.value = False

        gnvp3.print_analysis_options()

        polars_time: float = time.time()
        gnvp3.run()
        print(
            f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
        )
        polars: DataFrame | int = gnvp3.get_results()
        airplane.save()
        if isinstance(polars, int):
            continue
        if not DYNAMICS[airplane.name]:
            continue

        # # Dynamics
        # ### Define and Trim Plane
        try:
            unstick = State("Unstick", airplane, polars, EARTH_ISA)
        except Exception as error:
            print(error)
            continue

        epsilons = None

        unstick.add_all_pertrubations("Central", epsilons)
        unstick.get_pertrub()

        # Define Analysis for Pertrubations
        # 3 Pertrubations Serial
        # 4 Pertrubations Parallel
        # 5 Sesitivity Analysis Serial
        # 6 Sesitivity Analysis Parallel
        analysis = gnvp3.available_analyses_names()[4]  # Pertrubations PARALLEL
        print(f"Selecting Analysis: {analysis}")
        gnvp3.set_analyses(analysis)

        options = gnvp3.get_analysis_options(verbose=False)

        if options is None:
            raise ValueError("Options not set")
        # Set Options
        options.plane.value = airplane
        options.state.value = unstick
        options.environment.value = EARTH_ISA
        options.solver2D.value = "Xfoil"
        options.maxiter.value = MAXITER[airplane.name]
        options.timestep.value = TIMESTEP[airplane.name]
        options.u_freestream.value = unstick.trim["U"]
        options.angle.value = unstick.trim["AoA"]

        # Run Analysis
        gnvp3.print_analysis_options()

        pert_time: float = time.time()
        print("Running Pertrubations")
        gnvp3.run()
        print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

        # Get Results And Save
        _ = gnvp3.get_results()
        unstick.save()

        # Sensitivity ANALYSIS
        # ADD SENSITIVITY ANALYSIS

    # print time program took
    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
