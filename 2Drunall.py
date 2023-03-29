import numpy as np
import os
import matplotlib.pyplot as plt

from Airfoils import airfoil as af

from Software.Foil2Wake import runF2w as f2w
from Software.OpenFoam import runOpenFoam as of
from Software.Xfoil import runXFoil as xf
from Database import BASEFOIL2W, BASEOPENFOAM, DB2D


HOMEDIR = os.getcwd()

# Reynolds And Mach and AoA


def ms2mach(ms):
    return ms / 340.29


def Re(v, c, n):
    return (v * c) / n


chordMax = 0.18
chordMin = 0.11
umax = 30
umin = 5
ne = 1.56e-5

Machmin = ms2mach(10)
Machmax = ms2mach(30)
Remax = Re(umax, chordMax, ne)
Remin = Re(umin, chordMin, ne)
AoAmax = 15
AoAmin = -6
NoAoA = (AoAmax - AoAmin) * 2 + 1

angles = np.linspace(AoAmin, AoAmax, NoAoA)
Reynolds = np.logspace(np.log10(Remin), np.log10(Remax), 5, base=10)
Mach = np.linspace(Machmax, Machmin, 10)

MACH = Machmax

cleaning = False
calcF2W = True
calcOpenFoam = False
calcXFoil = True

# LOOP
airfoils = ["4415", "0008"]
for airfoil in airfoils:
    print(f"\nRunning airfoil {airfoil}\n")
    # # Get Airfoil
    airf = af.AirfoilData.NACA(airfoil, n_points=200)
    airf.accessDB(HOMEDIR, DB2D)
    # airf.plotAirfoil()
    for Reyn in Reynolds:
        print(
            f"#################################### {Reyn} ######################################")

        # Setup Case Dirs
        airf.reynCASE(Reyn)

        # Foil2Wake
        ftrip_low = {"pos": 0.1, "neg": 0.2}
        ftrip_up = {"pos": 0.2, "neg": 0.1}
        Ncrit = 9
        print("------- Running Foil2Wake -------")
        if cleaning == True:
            airf.cleanRes(f2w.removeResults, [
                          airf.REYNDIR, airf.HOMEDIR, angles])
        if calcF2W == True:
            f2wargs = [airf.REYNDIR, airf.HOMEDIR, Reyn, MACH,
                       ftrip_low, ftrip_up, angles, f"naca{airf.name}"]
            airf.setupSolver(
                f2w.setupF2W, [BASEFOIL2W, airf.HOMEDIR, airf.REYNDIR])
            airf.runSolver(f2w.runF2W, f2wargs)
        # airf.makePolars(f2w.makeCLCD2, "Foil2Wake", [
            # airf.REYNDIR, airf.HOMEDIR])

        # # Xfoil
        print("-------  Running Xfoil ------- ")
        if calcXFoil == True:
            xfargs = [airf.REYNDIR, HOMEDIR, Reyn, MACH,
                      min(angles), max(angles), 0.5, airf.selig.T]
            XRES = airf.makePolars(xf.runAndSave, "XFOIL", xfargs)

        # # OpenFoam
        os.chdir(airf.REYNDIR)
        maxITER = 800
        print("------- Running OpenFoam ------- ")
        if cleaning == True:
            airf.cleanRes(of.cleanOpenFoam, [HOMEDIR, airf.REYNDIR])
        if calcOpenFoam == True:
            ofSetupargs = [BASEOPENFOAM, airf.REYNDIR, airf.HOMEDIR,
                           airf.airfile, airf.fname, Reyn, MACH, angles]
            ofSetupkwargs = {"silent": True, "maxITER": maxITER}
            ofRunargs = [angles]
            airf.setupSolver(of.setupOpenFoam, ofSetupargs, ofSetupkwargs)
            airf.runSolver(of.runFoam, [airf.REYNDIR, airf.HOMEDIR, angles])
        airf.makePolars(of.makeCLCD, "OpenFoam", [
                        airf.REYNDIR, airf.HOMEDIR, angles])
print("########################################################################")
print("Program Terminated")
print("########################################################################")
