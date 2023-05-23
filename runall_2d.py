import os
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database import BASEFOIL2W
from ICARUS.Database import BASEOPENFOAM
from ICARUS.Database import DB2D
from ICARUS.Software.F2Wsection import runF2w as f2w
from ICARUS.Software.OpenFoam import run_open_foam as of
from ICARUS.Software.Xfoil import runXFoil as xf

HOMEDIR: str = os.getcwd()

# Reynolds And Mach and AoA


def ms2mach(speed_m_s: float) -> float:
    """Converts speed in m/s to mach number

    Args:
        speed_m_s (float): Speed in m/s

    Returns:
        float: Mach Number of speed
    """

    return speed_m_s / 340.29


def Re(velocity: float, char_length: float, viscosity: float) -> float:
    """Computes Reynolds number from velocity, characteristic length and viscosity

    Args:
        velocity (float)
        char_length (float)
        viscosity (float)

    Returns:
        float: Reynolds number
    """
    return (velocity * char_length) / viscosity


chord_max: float = 0.18
chord_min: float = 0.11
u_max: float = 30.0
u_min: float = 5.0
viscosity: float = 1.56e-5

mach_min: float = ms2mach(10)
mach_max: float = ms2mach(30)
reynolds_max: float = Re(u_max, chord_max, viscosity)
reynolds_min: float = Re(u_min, chord_min, viscosity)
aoa_max: float = 15
aoa_min: float = -6
num_of_angles: float = (aoa_max - aoa_min) * 2 + 1

angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(
    start=aoa_min,
    stop=aoa_max,
    num=num_of_angles,
)
reynolds: ndarray[Any, dtype[floating[Any]]] = np.logspace(
    start=np.log10(reynolds_min),
    stop=np.log10(reynolds_max),
    num=5,
    base=10,
)
mach: ndarray[Any, dtype[floating[Any]]] = np.linspace(mach_max, mach_min, 10)

MACH: float = mach_max

cleaning: bool = False
calcF2W: bool = True
calcOpenFoam: bool = False
calcXFoil: bool = False

# LOOP
airfoil_names: list[str] = ["4415", "0008"]
for airfoil_name in airfoil_names:
    print(f"\nRunning airfoil {airfoil_name}\n")
    # # Get Airfoil
    airfoil: AirfoilD = AirfoilD.naca(naca=airfoil_name, n_points=200)
    airfoil.access_db(HOMEDIR, DB2D)
    # airf.plotAirfoil()
    for reyn in reynolds:
        print(
            f"#################################### {reyn} ######################################",
        )

        # Setup Case Dirs
        airfoil.set_reynolds_case(reyn)

        # Foil2Wake
        if cleaning:
            airfoil.clean_results(
                f2w.remove_results,
                [airfoil.REYNDIR, airfoil.HOMEDIR, angles],
            )
        if calcF2W:
            ftrip_low: dict[str, float] = {"pos": 0.1, "neg": 0.2}
            ftrip_up: dict[str, float] = {"pos": 0.2, "neg": 0.1}
            Ncrit = 9
            print("------- Running Foil2Wake -------")
            f2wargs = [
                airfoil.REYNDIR,
                airfoil.HOMEDIR,
                reyn,
                MACH,
                ftrip_low,
                ftrip_up,
                angles,
                f"naca{airfoil.name}",
            ]
            airfoil.solver_setup(
                f2w.setup_f2w,
                [BASEFOIL2W, airfoil.HOMEDIR, airfoil.REYNDIR],
            )
            # airf.runSolver(f2w.runF2W, f2wargs)
            airfoil.make_polars(
                f2w.make_2d_polars_2,
                "Foil2Wake",
                [airfoil.REYNDIR, airfoil.HOMEDIR],
            )

        # # Xfoil
        if calcXFoil:
            print("-------  Running Xfoil ------- ")
            xfargs = [
                airfoil.REYNDIR,
                HOMEDIR,
                reyn,
                MACH,
                min(angles),
                max(angles),
                0.5,
                airfoil.selig.T,
            ]
            XRES: DataFrame = airfoil.make_polars(xf.run_and_save, "XFOIL", xfargs)

        # # OpenFoam
        os.chdir(airfoil.REYNDIR)
        maxITER = 800
        if cleaning:
            airfoil.clean_results(of.clean_open_foam, [HOMEDIR, airfoil.REYNDIR])
        if calcOpenFoam:
            print("------- Running OpenFoam ------- ")
            ofSetupargs = [
                BASEOPENFOAM,
                airfoil.REYNDIR,
                airfoil.HOMEDIR,
                airfoil.airfile,
                airfoil.file_name,
                reyn,
                MACH,
                angles,
            ]
            ofSetupkwargs = {"silent": True, "maxITER": maxITER}
            ofRunargs = [angles]
            airfoil.solver_setup(of.setup_open_foam, ofSetupargs, ofSetupkwargs)
            airfoil.solver_run(
                of.run_multiple_angles,
                [airfoil.REYNDIR, airfoil.HOMEDIR, angles],
            )
            airfoil.make_polars(
                of.make_polars,
                "OpenFoam",
                [airfoil.REYNDIR, airfoil.HOMEDIR, angles],
            )
print("########################################################################")
print("Program Terminated")
print("########################################################################")
