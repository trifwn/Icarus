"""Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.

THIS PROGRAM USES THE LSPT SOLVER Developed in this project.
"""

import os
import time
from typing import Any

import numpy as np
from Planes.hermes import hermes

# CHANGE THIS TO YOUR DATABASE FOLDER
from ICARUS import INSTALL_DIR
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.environment import EARTH_ISA
from ICARUS.flight_dynamics import State
from ICARUS.solvers.Icarus_LSPT import LSPT
from ICARUS.vehicle import Airplane

database_folder = os.path.join(
    INSTALL_DIR,
    "Data",
)


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # Load the database
    _ = Database(database_folder)

    # # Get Plane
    planes: list[Airplane] = []
    UINF: dict[str, Any] = {}
    ALTITUDE: dict[str, Any] = {}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, Any] = {}

    her: Airplane = hermes(name="plane_1")
    UINF["plane_1"] = 20
    ALTITUDE["plane_1"] = 0
    TEMPERATURE["plane_1"] = 273 + 15
    planes.append(her)

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(
            ALTITUDE[airplane.name],
            TEMPERATURE[airplane.name],
        )
        print(EARTH_ISA)
        state = State(
            name="Unstick",
            airplane=airplane,
            environment=EARTH_ISA,
            u_freestream=UINF[airplane.name],
        )
        # # Get Solver
        lspt = LSPT()

        # ## AoA Run
        # 0: Angles Sequential

        analysis = lspt.aseq
        inputs = analysis.get_analysis_input(verbose=False)
        solver_parameters = lspt.get_solver_parameters()

        AOA_MIN = -5
        AOA_MAX = 6
        NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
        angles: FloatArray = np.linspace(
            AOA_MIN,
            AOA_MAX,
            NO_AOA,
        )

        inputs.plane = airplane
        inputs.state = state
        inputs.angles = angles

        solver_parameters.ground_effect = True
        solver_parameters.wake_type = "TE-Geometrical"

        polars_time: float = time.time()
        lspt.execute(
            analysis=analysis,
            inputs=inputs,
            solver_parameters=solver_parameters,
        )
        print(
            f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
        )
        airplane.save()

    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
