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
from Planes.hermes import hermes
from Planes.hermes_wing_only import hermes_main_wing
from Planes.wing_variations import wing_var_chord_offset

from ICARUS.Core.struct import Struct
from ICARUS.Database import XFLRDB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import EARTH
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Software.GenuVP.gnvp7 import get_gnvp7
from ICARUS.Software.XFLR5.parser import parse_xfl_project
from ICARUS.Software.XFLR5.polars import read_polars_2d
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

    planes.append(wing_var_chord_offset(airfoils, "orthogonal_7", [0.159, 0.159], 0.0))

    planes.append(
        wing_var_chord_offset(airfoils, "orthSweep_7", [0.32, 0.32], 0.001),
    )

    planes.append(wing_var_chord_offset(airfoils, "taperSweep_7", [0.159, 0.072], 0.2))

    planes.append(wing_var_chord_offset(airfoils, "taper_7", [0.159, 0.072], 0.0))

    planes.append(hermes(airfoils=airfoils, name='hermes_7'))

    timestep: dict[str, float] = {
        "orthogonal_7": 1e-3,
        "orthogonalSweep_7": 1e-3,
        "taperSweep_7": 1e-3,
        "taper_7": 1e-3,
        "atlas_7": 1e-3,
        "hermes_7": 1e-3,
    }
    maxiter: dict[str, int] = {
        "orthogonal_7": 100,
        "orthogonalSweep_7": 100,
        "taperSweep_7": 400,
        "taper_7": 400,
        "atlas_7": 400,
        "hermes_7": 400,
    }
    filename: str = "Data/XFLR5/atlas.xml"
    atlas: Airplane = parse_xfl_project(filename)
    atlas.name = "atlas_7"
    atlas.visualize()
    planes.append(atlas)

    for airplane in planes:
        print("--------------------------------------------------")
        print(f"Running {airplane.name}")
        print("--------------------------------------------------")

        # # Import Enviroment
        print(EARTH)

        # # Get Solver
        gnvp3: Solver = get_gnvp7(db)

        # ## AoA Run
        analysis: str = gnvp3.available_analyses_names()[2]  # ANGLES PARALLEL
        gnvp3.set_analyses(analysis)
        options: Struct = gnvp3.get_analysis_options(verbose=False)
        solver_parameters: Struct = gnvp3.get_solver_parameters()

        AOA_MIN = -6
        AOA_MAX = 10
        NO_AOA: int = (AOA_MAX - AOA_MIN) + 1
        angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(
            AOA_MIN,
            AOA_MAX,
            NO_AOA,
        )
        UINF = 20
        # airplane.define_dynamic_pressure(UINF, EARTH.air_density)

        options.plane.value = airplane
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "Foil2Wake"
        options.maxiter.value = maxiter[airplane.name]
        options.timestep.value = timestep[airplane.name]
        options.u_freestream.value = UINF
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

        # # Dynamics
        # ### Define and Trim Plane
        try:
            unstick = State("Unstick", airplane, polars, EARTH)
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
        analysis = gnvp3.available_analyses_names()[3]  # Pertrubations PARALLEL
        print(f"Selecting Analysis: {analysis}")
        gnvp3.set_analyses(analysis)

        options = gnvp3.get_analysis_options(verbose=False)

        if options is None:
            raise ValueError("Options not set")
        # Set Options
        options.plane.value = airplane
        options.state.value = unstick
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "Foil2Wake"
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
