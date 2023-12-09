"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.

THIS PROGRAM USES THE LSPT SOLVER Developed in this project.
"""
import time
from typing import Any

import numpy as np
from Vehicles.Planes.e190_cruise import e190_cruise
from Vehicles.Planes.e190_takeoff import e190_takeoff_generator
from Vehicles.Planes.hermes import hermes
from Vehicles.Planes.wing_variations import wing_var_chord_offset

from ICARUS.Computation.Solvers.Icarus_LSPT.wing_lspt import get_lspt
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Computation.Solvers.XFLR5.parser import parse_xfl_project
from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Database import EXTERNAL_DB
from ICARUS.Environment.definition import EARTH_ISA
from ICARUS.Vehicle.plane import Airplane


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # DB CONNECTION
    read_polars_2d(EXTERNAL_DB)

    # # Get Plane
    planes: list[Airplane] = []
    UINF: dict[str, Any] = {}
    ALTITUDE: dict[str, Any] = {}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, Any] = {}

    from ICARUS.Computation.Solvers.XFLR5.parser import parse_xfl_project

    filename: str = "Data/3D_Party/plane_titos.xml"
    airplane = parse_xfl_project(filename)
    planes.append(airplane)

    embraer_cr: Airplane = e190_cruise(name="plane_1")
    UINF["plane_1"] = 20
    ALTITUDE["plane_1"] = 0
    TEMPERATURE["plane_1"] = 273 + 15
    planes.append(embraer_cr)

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Environment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(ALTITUDE[airplane.name], TEMPERATURE[airplane.name])
        print(EARTH_ISA)

        # # Get Solver
        lspt: Solver = get_lspt()

        # ## AoA Run
        # 0: Angles Sequential

        analysis: str = lspt.available_analyses_names()[0]
        lspt.set_analyses(analysis)
        options: Struct = lspt.get_analysis_options(verbose=False)
        solver_parameters: Struct = lspt.get_solver_parameters()

        AOA_MIN = -5
        AOA_MAX = 6
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
        options.u_freestream.value = UINF[airplane.name]
        options.angles.value = angles

        solver_parameters.Ground_Effect.value = True
        solver_parameters.Wake_Geom_Type.value = "TE-Geometrical"

        lspt.print_analysis_options()

        polars_time: float = time.time()
        lspt.run()
        print(
            f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
        )
        airplane.save()

    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
