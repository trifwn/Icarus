import os
import time

import numpy as np

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Computation.Solvers.OpenFoam.files.setup_case import MeshType
from ICARUS.Computation.Solvers.solver import Solver
from ICARUS.Computation.Solvers.XFLR5.polars import read_polars_2d
from ICARUS.Core.struct import Struct
from ICARUS.Core.types import FloatArray
from ICARUS.Core.units import calc_mach
from ICARUS.Core.units import calc_reynolds
from ICARUS.Database import DB
from ICARUS.Database import EXTERNAL_DB


def main() -> None:
    """Main function to run multiple airfoil simulations"""
    start_time: float = time.time()

    # SETUP DB CONNECTION
    read_polars_2d(EXTERNAL_DB)

    # RUN SETUP
    calcF2W: bool = False
    calcOpenFoam: bool = False  # True
    calcXFoil: bool = True
    print("Running:")
    print(f"\tFoil2Wake section: {calcF2W}")
    print(f"\tXfoil: {calcXFoil}")
    print(f"\tOpenfoam: {calcOpenFoam}")

    # airfoil SETUP
    airfoils: list[Airfoil] = []

    airfoil_names: list[str] = ["0015", "0008", "0012", "2412", "4415"]
    # airfoil_names: list[str] = ["2412", "4415"]
    # Load From DB
    db_airfoils: Struct = DB.foils_db.set_available_airfoils()
    for airfoil_name in airfoil_names:
        try:
            airfoils.append(db_airfoils[airfoil_name])
        except KeyError:
            print(f"Airfoil {airfoil_name} not found in database")
            print("Trying to Generate it")
            airfoils.append(Airfoil.naca(naca=airfoil_name, n_points=400))

    # # Load From File
    # for airfoil_name in airfoil_names:
    #     airfoils.append(airfoil.naca(naca=airfoil_name, n_points=200))

    # naca64418: Airfoil = Airfoil.load_from_file(os.path.join(XFLRDB, "NACA64418", "naca64418.dat"))
    # airfoils.append(naca64418)

    # naca64418_fl: Airfoil = naca64418.flap_airfoil(0.75, 1.3, 35)
    # airfoils.append(naca64418_fl)

    # PARAMETERS FOR ESTIMATION
    chord_max: float = 0.4
    chord_min: float = 0.4
    u_max: float = 40
    u_min: float = 5
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
        num=5,
    )

    # ANGLE OF ATTACK SETUP
    aoa_min: float = -5
    aoa_max: float = 12
    num_of_angles: int = int((aoa_max - aoa_min) * 2 + 1)
    angles: FloatArray = np.linspace(
        start=aoa_min,
        stop=aoa_max,
        num=num_of_angles,
    )

    # Transition to turbulent Boundary Layer
    ftrip_up: dict[str, float] = {"pos": 0.4, "neg": 0.4}
    ftrip_low: dict[str, float] = {"pos": 0.1, "neg": 0.1}
    Ncrit = 9

    #   ############################## START LOOP ###########################################
    for airfoil in airfoils:
        print(airfoil.name)
        airfoil_stime: float = time.time()
        print(f"\nRunning airfoil {airfoil.name}\n")
        # # Get airfoil
        airfoil.plot()

        # Foil2Wake
        if calcF2W:
            f2w_stime: float = time.time()
            from ICARUS.Computation.Solvers.Foil2Wake.f2w_section import get_f2w_section

            f2w_s: Solver = get_f2w_section()

            analysis: str = f2w_s.available_analyses_names()[0]  # ANGLES PARALLEL
            f2w_s.set_analyses(analysis)
            f2w_options: Struct = f2w_s.get_analysis_options(verbose=True)
            f2w_solver_parameters: Struct = f2w_s.get_solver_parameters()

            # Set Options
            f2w_options.airfoil.value = airfoil
            f2w_options.reynolds.value = reynolds
            f2w_options.mach.value = MACH
            f2w_options.angles.value = angles
            f2w_s.print_analysis_options()

            f2w_solver_parameters.f_trip_upper.value = ftrip_up["pos"]
            f2w_solver_parameters.f_trip_low.value = ftrip_low["pos"]
            f2w_solver_parameters.Ncrit.value = Ncrit
            f2w_solver_parameters.max_iter.value = 400
            f2w_solver_parameters.boundary_layer_solve_time.value = 399  # IF STEADY SHOULD BE 1 LESS THAN MAX ITER
            f2w_solver_parameters.timestep.value = 0.1

            f2w_s.run()

            _ = f2w_s.get_results()
            f2w_etime: float = time.time()
            print(f"Foil2Wake completed in {f2w_etime - f2w_stime} seconds")
        # XFoil
        if calcXFoil:
            xfoil_stime: float = time.time()
            from ICARUS.Computation.Solvers.Xfoil.xfoil import get_xfoil

            xfoil: Solver = get_xfoil()

            # Import Analysis
            # 0) Sequential Angle run for multiple reynolds in parallel,
            # 1) Sequential Angle run for multiple reynolds in serial,
            # 2) Sequential Angle run for multiple reynolds in parallel with zeroing of the boundary layer between angles,
            # 3) Sequential Angle run for multiple reynolds in serial with zeroing of the boundary layer between angles,
            analysis = xfoil.available_analyses_names()[0]  # Run
            xfoil.set_analyses(analysis)

            # Get Options
            xfoil_options: Struct = xfoil.get_analysis_options(verbose=True)
            xfoil_solver_parameters: Struct = xfoil.get_solver_parameters()

            # Set Options
            xfoil_options.airfoil.value = airfoil
            xfoil_options.reynolds.value = reynolds
            xfoil_options.mach.value = MACH
            xfoil_options.max_aoa.value = aoa_max
            xfoil_options.min_aoa.value = aoa_min
            xfoil_options.aoa_step.value = 0.5
            # xfoil_options.angles.value = angles  # For options 2 and 3

            xfoil.print_analysis_options()
            # Set Solver Options
            xfoil_solver_parameters.max_iter.value = 1000

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
                from ICARUS.Computation.Solvers.OpenFoam.open_foam import get_open_foam

                open_foam: Solver = get_open_foam()

                # Import Analysis
                analysis = open_foam.available_analyses_names()[0]  # Run
                open_foam.set_analyses(analysis)

                # Get Options
                of_options: Struct = open_foam.get_analysis_options(verbose=True)
                of_solver_parameters: Struct = open_foam.get_solver_parameters()

                # Set Options
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

        # from ICARUS.Visualization.airfoil.db_polars import plot_airfoil_polars

        # DB.foils_db.load_data()
        # solvers = []
        # if calcF2W:
        #     solvers += ["Foil2Wake"]
        # if calcXFoil:
        #     solvers += ["Xfoil"]
        # if calcOpenFoam:
        #     solvers += ["OpenFoam"]

        # axs, fig = plot_airfoil_polars(
        #     airfoil_name=airfoil.name,
        #     solvers=solvers,
        #     size=(10, 9),
        # )
    #   ############################### END LOOP ##############################################

    end_time = time.time()
    print(f"Total time: {end_time - start_time}")
    print("########################################################################")
    print("Program Terminated")
    print("########################################################################")


if __name__ == "__main__":
    # # parse command line options
    # # Options are -n for number of processors
    # from ICARUS import CPU_TO_USE

    # from optparse import OptionParser
    # parser = OptionParser()
    # parser.add_option("-n", "--num_proc", type="int", dest="num_proc", default=CPU_TO_USE,
    #                   help="Number of processors to use")
    # (options, args) = parser.parse_args()
    # num_proc = options.num_proc

    # # if num proc is specified, set it
    # if num_proc > 0:
    #     CPU_TO_USE = num_proc

    main()
