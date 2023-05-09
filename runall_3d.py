"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import time

import numpy as np

from Data.Planes.wing_variations import wing_var_chord_offset
from ICARUS.Core.struct import Struct
from ICARUS.Database import XFLRDB
from ICARUS.Database.Database_2D import Database_2D
from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import EARTH
from ICARUS.Flight_Dynamics.dynamic_plane import Dynamic_Airplane as dp
from ICARUS.Software.GenuVP3.gnvp3 import get_gnvp3
from ICARUS.Software.XFLR5.polars import readPolars2D
from ICARUS.Vehicle.plane import Airplane

# from Data.Planes.hermes import hermes
# from Data.Planes.hermes_wing_only import hermes_main_wing
# # MODULES


def main() -> None:
    """Main function to run the simulations."""
    start_time: float = time.time()

    # # DB CONNECTION
    db = DB()
    foildb: Database_2D = db.foilsDB
    foildb.loadData()
    readPolars2D(foildb, XFLRDB)
    airfoils = foildb.getAirfoils()

    # # Get Plane
    planes: list[Airplane] = []
    planes.append(wing_var_chord_offset(airfoils, "orthogonal", [0.159, 0.159], 0.0))

    planes.append(
        wing_var_chord_offset(airfoils, "orthogonalSweep", [0.159, 0.159], 0.2),
    )

    planes.append(wing_var_chord_offset(airfoils, "taperSweep", [0.159, 0.072], 0.2))

    planes.append(wing_var_chord_offset(airfoils, "taper", [0.159, 0.072], 0.0))

    timestep = {
        "orthogonal": 1e-3,
        "orthogonalSweep": 1e-3,
        "taperSweep": 1e-3,
        "taper": 1e-3,
    }
    maxiter = {
        "orthogonal": 400,
        "orthogonalSweep": 400,
        "taperSweep": 400,
        "taper": 400,
    }

    for airplane in planes:
        print(airplane.name)

        # # Import Enviroment
        print(EARTH)

        # # Get Solver
        gnvp3 = get_gnvp3(db)

        # ## AoA Run
        analysis = gnvp3.getAvailableAnalyses()[2]  # ANGLES PARALLEL
        gnvp3.setAnalysis(analysis)
        options = gnvp3.getOptions(verbose=True)

        AOA_MIN = -6
        AOA_MAX = 8
        NO_AOA = (AOA_MAX - AOA_MIN) + 1
        angles = np.linspace(AOA_MIN, AOA_MAX, NO_AOA)
        UINF = 20
        airplane.defineSim(UINF, EARTH.air_density)

        options.plane.value = airplane
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "XFLR"
        options.maxiter.value = maxiter[airplane.name]
        options.timestep.value = timestep[airplane.name]
        options.u_freestream.value = UINF
        options.angles.value = angles

        gnvp3.printOptions()

        polars_time = time.time()
        gnvp3.run()
        print(
            f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
        )
        polars = gnvp3.getResults()
        airplane.save()

        # # Dynamics

        # ### Define and Trim Plane
        try:
            dyn = dp(airplane, polars)
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

        dyn.allPerturb("Central", epsilons)
        dyn.get_pertrub()

        # Define Analysis for Pertrubations
        analysis: str = gnvp3.getAvailableAnalyses()[4]  # Pertrubations PARALLEL
        print(f"Selecting Analysis: {analysis}")
        gnvp3.setAnalysis(analysis)

        options: Struct = gnvp3.getOptions(verbose=True)

        if options is None:
            raise ValueError("Options not set")
        # Set Options
        dyn.defineSim(dyn.trim["U"], EARTH.air_density)
        options.plane.value = dyn
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "XFLR"
        options.maxiter.value = maxiter[airplane.name]
        options.timestep.value = timestep[airplane.name]
        options.u_freestream.value = dyn.trim["U"]
        options.angles.value = dyn.trim["AoA"]

        # Run Analysis
        gnvp3.printOptions()

        pert_time: float = time.time()
        print("Running Pertrubations")
        gnvp3.run()
        print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

        # Get Results And Save
        _ = gnvp3.getResults()
        dyn.save()

        # Sensitivity ANALYSIS
        # ADD SENSITIVITY ANALYSIS

    # print time program took
    print("PROGRAM TERMINATED")
    print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
