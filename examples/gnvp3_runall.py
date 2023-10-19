"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import time
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame
from Planes.e190_cruise import e190_cruise
from Planes.e190_takeoff import e190_takeoff_generator
from Planes.hermes import hermes
from Planes.wing_variations import wing_var_chord_offset

from ICARUS.Core.struct import Struct
from ICARUS.Database import XFLRDB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import EARTH_ISA
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Input_Output.XFLR5.parser import parse_xfl_project
from ICARUS.Input_Output.XFLR5.polars import read_polars_2d
from ICARUS.Solvers.Airplane.gnvp3 import get_gnvp3
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Workers.solver import Solver


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # DB CONNECTION
    db = DB()
    db.load_data()
    airfoils = db.foilsDB.airfoils
    foildb: Database_2D = db.foilsDB
    foildb.load_data()
    read_polars_2d(foildb, XFLRDB)

    # # Get Plane
    planes: list[Airplane] = []

    timestep: dict[str, float] = {
        "e190_takeoff_3": 10,
        "e190_cruise_3": 10,
    }
    maxiter: dict[str, int] = {
        "e190_takeoff_3": 50,
        "e190_cruise_3": 50,
    }

    UINF: dict[str, float] = {
        "e190_takeoff_3": 20,
        "e190_cruise_3": 232,
    }

    ALTITUDE: dict[str, int] = {"e190_cruise_3": 12000, "e190_takeoff_3": 0}

    # OUR ATMOSPHERIC MODEL IS NOT COMPLETE TO HANDLE TEMPERATURE VS ALTITUDE
    TEMPERATURE: dict[str, int] = {
        "e190_cruise_3": 273 - 50,
        "e190_takeoff_3": 273 + 15,
    }
    DYNAMICS: dict[str, float] = {
        "e190_takeoff_3": False,
        "e190_cruise_3": False,
    }

    embraer_to: Airplane = e190_takeoff_generator(name="e190_takeoff_3")
    embraer_cr: Airplane = e190_cruise(name="e190_cruise_3")

    # embraer.visualize()
    planes.append(embraer_to)
    planes.append(embraer_cr)

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Enviroment
        EARTH_ISA._set_pressure_from_altitude_and_temperature(ALTITUDE[airplane.name], TEMPERATURE[airplane.name])
        print(EARTH_ISA)

        # # Get Solver
        gnvp3: Solver = get_gnvp3(db)

        # ## AoA Run
        # 0: Single Angle of Attack (AoA) Run
        # 1: Angles Sequential
        # 2: Angles Parallel

        analysis: str = gnvp3.available_analyses_names()[2]
        gnvp3.set_analyses(analysis)
        options: Struct = gnvp3.get_analysis_options(verbose=False)
        solver_parameters: Struct = gnvp3.get_solver_parameters()

        AOA_MIN = -5
        AOA_MAX = 6
        NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
        angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            AOA_MIN,
            AOA_MAX,
            NO_AOA,
        )

        airplane.define_dynamic_pressure(UINF[airplane.name], EARTH_ISA.air_density)

        options.plane.value = airplane
        options.environment.value = EARTH_ISA
        options.db.value = db
        options.solver2D.value = "Xfoil"
        options.maxiter.value = maxiter[airplane.name]
        options.timestep.value = timestep[airplane.name]
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
        options.db.value = db
        options.solver2D.value = "Xfoil"
        options.maxiter.value = maxiter[airplane.name]
        options.timestep.value = timestep[airplane.name]
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
