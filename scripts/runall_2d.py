import os
from typing import Any

import numpy as np
from numpy import dtype
from numpy import floating
from numpy import ndarray

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


chordMax: float = 0.18
chordMin: float = 0.11
umax: float = 30.0
umin: float = 5.0
viscosity: float = 1.56e-5

Machmin: float = ms2mach(10)
Machmax: float = ms2mach(30)
Remax: float = Re(umax, chordMax, viscosity)
Remin: float = Re(umin, chordMin, viscosity)
AoAmax: float = 15
aoa_min: float = -6
NoAoA: int = (AoAmax - aoa_min) * 2 + 1

angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(aoa_min, AoAmax, NoAoA)
reynolds: ndarray[Any, dtype[floating]] = np.logspace(
    np.log10(Remin),
    np.log10(Remax),
    5,
    base=10,
)
mach: ndarray[Any, dtype[floating[Any]]] = np.linspace(Machmax, Machmin, 10)

MACH: float = Machmax

cleaning: bool = False
calcF2W: bool = True
calcOpenFoam: bool = False
calcXFoil: bool = False

# LOOP
airfoil_names: list[str] = ["4415", "0008"]
for airfoil_name in airfoil_names:
    print(f"\nRunning airfoil {airfoil_name}\n")
    # # Get Airfoil
    airfoil: AirfoilD = AirfoilD.NACA(naca=airfoil_name, n_points=200)
    airfoil.accessDB(HOMEDIR, DB2D)
    # airf.plotAirfoil()
    for reyn in reynolds:
        print(
            f"#################################### {reyn} ######################################",
        )

        # Setup Case Dirs
        airfoil.reynCASE(reyn)

        # Foil2Wake
        if cleaning:
            airfoil.cleanRes(
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
            airfoil.setupSolver(
                f2w.setup_f2w,
                [BASEFOIL2W, airfoil.HOMEDIR, airfoil.REYNDIR],
            )
            # airf.runSolver(f2w.runF2W, f2wargs)
            airfoil.makePolars(
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
            XRES = airfoil.makePolars(xf.run_and_save, "XFOIL", xfargs)

        # # OpenFoam
        os.chdir(airfoil.REYNDIR)
        maxITER = 800
        if cleaning:
            airfoil.cleanRes(of.clean_open_foam, [HOMEDIR, airfoil.REYNDIR])
        if calcOpenFoam:
            print("------- Running OpenFoam ------- ")
            ofSetupargs = [
                BASEOPENFOAM,
                airfoil.REYNDIR,
                airfoil.HOMEDIR,
                airfoil.airfile,
                airfoil.fname,
                reyn,
                MACH,
                angles,
            ]
            ofSetupkwargs = {"silent": True, "maxITER": maxITER}
            ofRunargs = [angles]
            airfoil.setupSolver(of.setup_open_foam, ofSetupargs, ofSetupkwargs)
            airfoil.runSolver(
                of.run_multiple_angles,
                [airfoil.REYNDIR, airfoil.HOMEDIR, angles],
            )
            airfoil.makePolars(
                of.make_polars,
                "OpenFoam",
                [airfoil.REYNDIR, airfoil.HOMEDIR, angles],
            )
print("########################################################################")
print("Program Terminated")
print("########################################################################")
