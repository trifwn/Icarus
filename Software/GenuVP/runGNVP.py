import os
import numpy as np

from . import filesGNVP as fgnvp


def GNVPexe(HOMEDIR, ANGLEDIR):
    os.chdir(ANGLEDIR)
    os.system("./gnvp < input > gnvp.out")
    # os.system(f"cat LOADS_aer.dat >>  res.dat")
    os.chdir(HOMEDIR)


def runGNVP(plane, GENUBASE, polars, solver, Uinf, angles):
    CASEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    bodies = []
    airMovement = airMov()

    plane.save()
    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for angle in angles:
        print(f"Running Angles {angle}")
        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1] + "/"
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"

        ANGLEDIR = f"{CASEDIR}/{folder}"
        os.system(f"mkdir -p {ANGLEDIR}")

        params = setParams(len(bodies), len(airfoils), Uinf, angle)

        fgnvp.makeInput(ANGLEDIR, HOMEDIR, GENUBASE, airMovement,
                        bodies, params, airfoils, polars, solver)
        GNVPexe(HOMEDIR, ANGLEDIR)
    fgnvp.makePolar(CASEDIR, HOMEDIR)


def airMov():
    airMovement = {
        'alpha_s': 0.,
        'alpha_e': 0.,
        'beta_s': 0.,
        'beta_e': 0.,
        'phi_s': 0.,
        'phi_e': 0.
    }
    return airMovement


def setParams(nBodies, nAirfoils, Uinf, WindAngle):
    params = {
        "nBods": nBodies,
        "nBlades": nAirfoils,
        "maxiter": 50,
        "timestep": 10,
        "Uinf": [Uinf * np.cos(WindAngle*np.pi/180), 0.0, Uinf * np.sin(WindAngle*np.pi/180)],
        "rho": 1.225,
        "visc": 0.0000156,
    }
    return params


def makeSurfaceDict(surf, idx):
    s = {
        'NB': idx,
        "NACA": 4415,
        "name": surf.name,
        'bld': f'{surf.name}.bld',
        'cld': f'{surf.airfoil.name}.cld',
        'NNB': surf.N,
        'NCWB': surf.M,
        "x_0": surf.Origin[0],
        "y_0": surf.Origin[1],
        "z_0": surf.Origin[2],
        "pitch": surf.Orientation[0],
        "cone": surf.Orientation[1],
        "wngang": surf.Orientation[2],
        "x_end": surf.Origin[0] + surf.xoff[-1],
        "y_end": surf.Origin[1] + surf.Dspan[-1],
        "z_end": surf.Origin[2] + surf.Ddihedr[-1],
        "Root_chord": surf.chord[0],
        "Tip_chord": surf.chord[-1]
    }
    return s
