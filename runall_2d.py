import os

import numpy as np
from numpy import dtype, ndarray, floating
from typing import Any

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Database import BASEFOIL2W
from ICARUS.Database import BASEOPENFOAM
from ICARUS.Database import DB2D
from ICARUS.Software.F2Wsection import runF2w as f2w
from ICARUS.Software.OpenFoam import runOpenFoam as of
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

Machmin = ms2mach(10)
Machmax: float = ms2mach(30)
Remax: float = Re(umax, chordMax, viscosity)
Remin: float = Re(umin, chordMin, viscosity)
AoAmax: float = 15
AoAmin: float = -6
NoAoA: int = (AoAmax - AoAmin) * 2 + 1

angles: ndarray[Any, dtype[floating[Any]]] = np.linspace(AoAmin, AoAmax, NoAoA)
Reynolds: ndarray[Any, dtype[floating[Any]]] = np.logspace(np.log10(Remin), np.log10(Remax), 5, base=10)
Mach: ndarray[Any, dtype[floating[Any]]] = np.linspace(Machmax, Machmin, 10)

MACH: float = Machmax

cleaning: bool = False
calcF2W: bool = True
calcOpenFoam: bool = False
calcXFoil: bool = False

# LOOP
airfoils: list[str] = ["4415", "0008"]
for airf in airfoils:
    print(f"\nRunning airfoil {airf}\n")
    # # Get Airfoil
    airf: AirfoilD = AirfoilD.NACA(airf, n_points=200)
    airf.accessDB(HOMEDIR, DB2D)
    # airf.plotAirfoil()
    for Reyn in Reynolds:
        print(
            f"#################################### {Reyn} ######################################",
        )

        # Setup Case Dirs
        airf.reynCASE(Reyn)

        # Foil2Wake
        if cleaning:
            airf.cleanRes(f2w.removeResults, [airf.REYNDIR, airf.HOMEDIR, angles])
        if calcF2W:
            ftrip_low = {"pos": 0.1, "neg": 0.2}
            ftrip_up = {"pos": 0.2, "neg": 0.1}
            Ncrit = 9
            print("------- Running Foil2Wake -------")
            f2wargs = [
                airf.REYNDIR,
                airf.HOMEDIR,
                Reyn,
                MACH,
                ftrip_low,
                ftrip_up,
                angles,
                f"naca{airf.name}",
            ]
            airf.setupSolver(f2w.setupF2W, [BASEFOIL2W, airf.HOMEDIR, airf.REYNDIR])
            # airf.runSolver(f2w.runF2W, f2wargs)
            airf.makePolars(f2w.makeCLCD2, "Foil2Wake", [airf.REYNDIR, airf.HOMEDIR])

        # # Xfoil
        if calcXFoil:
            print("-------  Running Xfoil ------- ")
            xfargs = [
                airf.REYNDIR,
                HOMEDIR,
                Reyn,
                MACH,
                min(angles),
                max(angles),
                0.5,
                airf.selig.T,
            ]
            XRES = airf.makePolars(xf.runAndSave, "XFOIL", xfargs)

        # # OpenFoam
        os.chdir(airf.REYNDIR)
        maxITER = 800
        if cleaning:
            airf.cleanRes(of.cleanOpenFoam, [HOMEDIR, airf.REYNDIR])
        if calcOpenFoam:
            print("------- Running OpenFoam ------- ")
            ofSetupargs = [
                BASEOPENFOAM,
                airf.REYNDIR,
                airf.HOMEDIR,
                airf.airfile,
                airf.fname,
                Reyn,
                MACH,
                angles,
            ]
            ofSetupkwargs = {"silent": True, "maxITER": maxITER}
            ofRunargs = [angles]
            airf.setupSolver(of.setupOpenFoam, ofSetupargs, ofSetupkwargs)
            airf.runSolver(of.runFoam, [airf.REYNDIR, airf.HOMEDIR, angles])
            airf.makePolars(
                of.makeCLCD,
                "OpenFoam",
                [airf.REYNDIR, airf.HOMEDIR, angles],
            )
print("########################################################################")
print("Program Terminated")
print("########################################################################")
