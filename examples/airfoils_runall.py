import os
import time
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.struct import Struct
from ICARUS.Core.Units import calc_mach
from ICARUS.Core.Units import calc_reynolds
from ICARUS.Database import XFLRDB
from ICARUS.Database.db import DB
from ICARUS.Input_Output.OpenFoam.filesOpenFoam import MeshType
from ICARUS.Input_Output.XFLR5.polars import read_polars_2d
from ICARUS.Workers.solver import Solver


def main() -> None:
    """Main function to run multiple airfoil simulations"""
    start_time: float = time.time()

    # SETUP DB CONNECTION
    db = DB()
    # db.load_data()
    read_polars_2d(db.foilsDB, XFLRDB)

    # RUN SETUP
    calcF2W: bool = True  # True
    calcOpenFoam: bool = False  # True
    calcXFoil: bool = True
    print("Running:")
    print(f"\tFoil2Wake section: {calcF2W}")
    print(f"\tXfoil: {calcXFoil}")
    print(f"\tOpenfoam: {calcOpenFoam}")

    # AIRFOIL SETUP
    airfoils: list[AirfoilD] = []

    # airfoil_names: list[str] = ["2412", "0015", "0008", "4415", "0012"]
    airfoil_names: list[str] = ["0008", "4415", "0012"]
    # Load From DB
    db_airfoils: Struct = db.foilsDB.set_available_airfoils()
    for airfoil_name in airfoil_names:
        try:
            airfoils.append(db_airfoils[airfoil_name])
        except KeyError:
            print(f"Airfoil {airfoil_name} not found in database")
            print("Trying to Generate it")
            airfoils.append(AirfoilD.naca(naca=airfoil_name, n_points=200))

    # # Load From File
    # for airfoil_name in airfoil_names:
    #     airfoils.append(AirfoilD.naca(naca=airfoil_name, n_points=200))

    # naca64418: AirfoilD = AirfoilD.load_from_file(os.path.join(XFLRDB, "NACA64418", 'naca64418.dat'))
    # airfoils.append(naca64418)

    # naca64418_fl: AirfoilD = naca64418.flap_airfoil(0.75, 1.3, 35)
    # airfoils.append(naca64418_fl)

    # PARAMETERS FOR ESTIMATION
    chord_max: float = 0.4
    chord_min: float = 0.1
    u_max: float = 100
    u_min: float = 10
    viscosity: float = 1.56e-5

    # MACH ESTIMATION
    mach_max: float = 0.085
    # mach_min: float = calc_mach(10, speed_of_sound)
    # mach: ndarray[Any, dtype[floating[Any]]] = np.linspace(mach_max, mach_min, 10)
    MACH: float = mach_max

    # REYNOLDS ESTIMATION
    reynolds_max: float = calc_reynolds(u_max, chord_max, viscosity)
    reynolds_min: float = calc_reynolds(u_min, chord_min, viscosity)
    reynolds: ndarray[Any, dtype[floating[Any]]] = np.logspace(
        start=np.log10(reynolds_min),
        stop=np.log10(reynolds_max),
        num=20,
        base=10,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -10
    aoa_max: float = 15
    num_of_angles: int = int((aoa_max - aoa_min) * 2 + 1)
    angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(
        start=aoa_min,
        stop=aoa_max,
        num=num_of_angles,
    )

    # Transition to turbulent Boundary Layer
    ftrip_up: dict[str, float] = {"pos": 0.02, "neg": 0.01}
    ftrip_low: dict[str, float] = {"pos": 0.01, "neg": 0.02}
    Ncrit = 9

    #   ############################## START LOOP ###########################################
    for airfoil in airfoils:
        print(airfoil.name)
        airfoil_stime: float = time.time()
        print(f"\nRunning airfoil {airfoil.name}\n")
        # # Get Airfoil
        # airf.plotAirfoil()

        # Foil2Wake
        if calcF2W:
            f2w_stime: float = time.time()
            from ICARUS.Solvers.Airfoil.f2w_section import get_f2w_section

            f2w_s: Solver = get_f2w_section(db)

            analysis: str = f2w_s.available_analyses_names()[0]  # ANGLES PARALLEL
            f2w_s.set_analyses(analysis)
            f2w_options: Struct = f2w_s.get_analysis_options(verbose=True)
            f2w_solver_parameters: Struct = f2w_s.get_solver_parameters()

            # Set Options
            f2w_options.db.value = db
            f2w_options.airfoil.value = airfoil
            f2w_options.reynolds.value = reynolds
            f2w_options.mach.value = MACH
            f2w_options.angles.value = angles
            f2w_s.print_analysis_options()

            f2w_solver_parameters.f_trip_upper.value = ftrip_up["pos"]
            f2w_solver_parameters.f_trip_low.value = ftrip_low["pos"]
            f2w_solver_parameters.Ncrit.value = Ncrit
            f2w_solver_parameters.max_iter.value = 100
            # f2w_solver_parameters.max_iter_bl.value = 300
            f2w_solver_parameters.timestep.value = 0.001

            f2w_s.run()

            _ = f2w_s.get_results()
            f2w_etime: float = time.time()
            print(f"Foil2Wake completed in {f2w_etime - f2w_stime} seconds")
        # XFoil
        if calcXFoil:
            xfoil_stime: float = time.time()
            from ICARUS.Solvers.Airfoil.xfoil import get_xfoil

            xfoil: Solver = get_xfoil(db)

            # Import Analysis
            analysis = xfoil.available_analyses_names()[3]  # Run
            xfoil.set_analyses(analysis)

            # Get Options
            xfoil_options: Struct = xfoil.get_analysis_options(verbose=True)
            xfoil_solver_parameters: Struct = xfoil.get_solver_parameters()

            # Set Options
            xfoil_options.db.value = db
            xfoil_options.airfoil.value = airfoil
            xfoil_options.reynolds.value = reynolds
            xfoil_options.mach.value = MACH
            # xfoil_options.max_aoa.value = aoa_max
            # xfoil_options.min_aoa.value = aoa_min
            # xfoil_options.aoa_step.value = 0.5  # (aoa_max - aoa_min) / (num_of_angles + 1)
            xfoil_options.angles.value = angles
            xfoil.print_analysis_options()
            # Set Solver Options
            xfoil_solver_parameters.max_iter.value = 10000

            xfoil_solver_parameters.Ncrit.value = Ncrit
            xfoil_solver_parameters.xtr.value = (ftrip_up["pos"], ftrip_low["pos"])
            xfoil_solver_parameters.print.value = False
            # xfoil.print_solver_options()

            # RUN and SAVE
            xfoil.run()
            xfoil_etime: float = time.time()
            print(f"XFoil completed in {xfoil_etime - xfoil_stime} seconds")

        # OpenFoam
        if calcOpenFoam:
            of_stime: float = time.time()
            for reyn in reynolds:
                print(f"Running OpenFoam for Re={reyn}")
                from ICARUS.Solvers.Airfoil.open_foam import get_open_foam

                open_foam: Solver = get_open_foam(db)

                # Import Analysis
                analysis = open_foam.available_analyses_names()[0]  # Run
                open_foam.set_analyses(analysis)

                # Get Options
                of_options: Struct = open_foam.get_analysis_options(verbose=True)
                of_solver_parameters: Struct = open_foam.get_solver_parameters()

                # Set Options
                of_options.db.value = db
                of_options.airfoil.value = airfoil
                of_options.angles.value = angles
                of_options.reynolds.value = reyn
                of_options.mach.value = MACH
                open_foam.print_analysis_options()

                # Set Solver Options
                of_solver_parameters.mesh_type.value = MeshType.structAirfoilMesher
                of_solver_parameters.max_iterations.value = 100
                of_solver_parameters.silent.value = False
                # xfoil.print_solver_options()

                # RUN
                open_foam.run()
            of_etime: float = time.time()
            print(f"OpenFoam completed in {of_etime - of_stime} seconds")

        airfoil_etime: float = time.time()
        print(
            f"Airfoil {airfoil.name} completed in {airfoil_etime - airfoil_stime} seconds",
        )
    #   ############################### END LOOP ##############################################

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    print("########################################################################")
    print("Program Terminated")
    print("########################################################################")


if __name__ == "__main__":
    main()
