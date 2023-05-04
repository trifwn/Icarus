"""
Module to run multiple 3D simulations for different aircrafts sequentially.
It computes the polars for each aircraft and then computes the dynamics.
It is also possible to do a pertubation analysis for each aircraft.
"""
import os
import time

import numpy as np

from Data.Planes.wing_variations import wing_var_chord_offset
from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Enviroment.definition import EARTH
from ICARUS.Flight_Dynamics.dyn_plane import dyn_Airplane as dp
from ICARUS.Software.GenuVP3.gnvp3 import get_gnvp3
from ICARUS.Software.XFLR5.polars import readPolars2D
# from Data.Planes.hermes import hermes
# from Data.Planes.hermes_wing_only import hermesMainWing
# # MODULES


def main():
    """Main function to run the simulations."""
    start_time = time.time()

    # # DB CONNECTION
    db = DB()
    foildb = db.foilsDB
    foildb.loadData()
    readPolars2D(foildb, XFLRDB)
    airfoils = foildb.getAirfoils()

    # # Get Plane
    planes = list()
    planes.append(wing_var_chord_offset(airfoils, "orthogonal", [0.159, 0.159], 0.0))

    planes.append(
        wing_var_chord_offset(airfoils, "orthogonalSweep", [0.159, 0.159], 0.2),
    )

    planes.append(wing_var_chord_offset(airfoils, "taperSweep", [0.159, 0.072], 0.2))

    planes.append(wing_var_chord_offset(airfoils, "taper", [0.159, 0.072], 0.0))

    timestep = {
        "orthogonal": 5e-2,
        "orthogonalSweep": 5e-2,
        "taperSweep": 5e-2,
        "taper": 5e-2,
    }
    maxiter = {
        "orthogonal": 5e-2,
        "orthogonalSweep": 5e-2,
        "taperSweep": 5e-2,
        "taper": 5e-2,
    }

    for ap in planes:
        print(ap.name)

        # # Import Enviroment
        print(EARTH)

        # # Get Solver
        gnvp3 = get_gnvp3(db)

        # ## AoA Run
        analysis = gnvp3.getAvailableAnalyses()[2]  # ANGLES PARALLEL
        gnvp3.setAnalysis(analysis)
        options = gnvp3.getOptions(analysis)

        AOA_MIN = -6
        AOA_MAX = 8
        NO_AOA = (AOA_MAX - AOA_MIN) + 1
        angles = np.linspace(AOA_MIN, AOA_MAX, NO_AOA)
        UINF = 20
        ap.defineSim(UINF, EARTH.AirDensity)

        options.plane.value = ap
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "XFLR"
        options.maxiter.value = maxiter[ap.name]
        options.timestep.value = timestep[ap.name]
        options.Uinf.value = UINF
        options.angles.value = angles

        gnvp3.printOptions()

        polars_time = time.time()
        gnvp3.run()
        print(
            f"Polars took : --- {time.time() - polars_time} seconds --- in Parallel Mode",
        )
        polars = gnvp3.getResults()
        ap.save()

        # # Dynamics

        # ### Define and Trim Plane
        try:
            dyn = dp(ap, polars)
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
        analysis = gnvp3.getAvailableAnalyses()[4]  # Pertrubations PARALLEL
        print(f"Selecting Analysis: {analysis}")
        gnvp3.setAnalysis(analysis)
        options = gnvp3.getOptions(analysis)

        # Set Options
        dyn.defineSim(dyn.trim["U"], EARTH.AirDensity)
        options.plane.value = dyn
        options.environment.value = EARTH
        options.db.value = db
        options.solver2D.value = "XFLR"
        options.maxiter.value = maxiter[ap.name]
        options.timestep.value = timestep[ap.name]
        options.Uinf.value = dyn.trim["U"]
        options.angles.value = dyn.trim["AoA"]

        # Run Analysis
        gnvp3.printOptions()

        pert_time = time.time()
        print("Running Pertrubations")
        gnvp3.run()
        print(f"Pertrubations took : --- {time.time() - pert_time} seconds ---")

        # Get Results And Save
        _ = gnvp3.getResults()
        dyn.save()

        ## Sensitivity ANALYSIS
        # print time it took
        print("PROGRAM TERMINATED")
        print(f"Execution took : --- {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    main()
