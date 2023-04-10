import os
import numpy as np

from . import filesGNVP as fgnvp


def GNVPexe(HOMEDIR, ANGLEDIR):
    os.chdir(ANGLEDIR)
    os.system("./gnvp3 < input > gnvp.out")
    # os.system(f"cat LOADS_aer.dat >>  res.dat")
    os.chdir(HOMEDIR)


def runGNVPangles(plane, GENUBASE, polars, solver2D, maxiter, timestep, Uinf, angles, dens=1.225):
    PLANEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    bodies = []
    movements = airMov(plane.surfaces, plane.CG,
                       plane.orientation, plane.disturbances)

    plane.defineSim(Uinf, dens)

    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for angle in angles:
        print(f"Running Angles {angle}")
        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1] + "_AoA/"
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "_AoA/"

        CASEDIR = f"{PLANEDIR}/{folder}"
        os.system(f"mkdir -p {CASEDIR}")

        params = setParams(len(bodies), len(airfoils), maxiter, timestep,
                           Uinf, angle, dens)

        fgnvp.makeInput(CASEDIR, HOMEDIR, GENUBASE, movements,
                        bodies, params, airfoils, polars, solver2D)
        GNVPexe(HOMEDIR, CASEDIR)


def runGNVPpertr(plane, GENUBASE, polars, solver2D, maxiter, timestep, Uinf, angle):
    PLANEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    bodies = []

    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for dst in plane.disturbances:
        movements = airMov(plane.surfaces, plane.CG,
                           plane.orientation, [dst])

        print(f"Running Case {dst.var} - {dst.amplitude}")
        # if make distubance folder
        if dst.var == "Trim":
            folder = "Trim/"
        elif dst.isPositive:
            folder = "p" + \
                str(dst.amplitude)[::-1].zfill(6)[::-1] + f"_{dst.var}/"
        else:
            folder = "m" + \
                str(dst.amplitude)[
                    ::-1].strip("-").zfill(6)[::-1] + f"_{dst.var}/"

        CASEDIR = f"{PLANEDIR}/Dynamics/{folder}/"
        os.system(f"mkdir -p {CASEDIR}")

        params = setParams(len(bodies), len(airfoils), maxiter, timestep,
                           Uinf, angle, dens=plane.dens)

        fgnvp.makeInput(CASEDIR, HOMEDIR, GENUBASE, movements,
                        bodies, params, airfoils, polars, solver2D)
        GNVPexe(HOMEDIR, CASEDIR)


def runGNVPsensitivity(plane, var, GENUBASE, polars, solver2D, maxiter, timestep, Uinf, angle):
    PLANEDIR = plane.CASEDIR
    HOMEDIR = plane.HOMEDIR
    airfoils = plane.airfoils
    bodies = []

    for i, surface in enumerate(plane.surfaces):
        bodies.append(makeSurfaceDict(surface, i))

    for dst in plane.sensitivity[var]:
        movements = airMov(plane.surfaces, plane.CG,
                           plane.orientation, [dst])

        print(f"Running Case {dst.var} - {dst.amplitude}")
        # if make distubance folder
        if dst.isPositive:
            folder = "p" + \
                str(dst.amplitude)[::-1].zfill(6)[::-1] + f"_{dst.var}/"
        else:
            folder = "m" + \
                str(dst.amplitude)[
                    ::-1].strip("-").zfill(6)[::-1] + f"_{dst.var}/"

        CASEDIR = f"{PLANEDIR}/Sensitivity_{dst.var}/{folder}/"
        os.system(f"mkdir -p {CASEDIR}")

        params = setParams(len(bodies), len(airfoils), maxiter, timestep,
                           Uinf, angle, dens=plane.dens)

        fgnvp.makeInput(CASEDIR, HOMEDIR, GENUBASE, movements,
                        bodies, params, airfoils, polars, solver2D)
        GNVPexe(HOMEDIR, CASEDIR)


def makePolar(CASEDIR, HOMEDIR):
    return fgnvp.makePolar(CASEDIR, HOMEDIR)


def logResults(DYNDIR, HOMEDIR):
    return fgnvp.logPertrub(DYNDIR, HOMEDIR)


def airMov(surfaces, CG, orientation, disturbances):
    movement = []
    for surface in surfaces:
        sequence = []
        for name, axis in [["pitch", 2], ["roll", 1], ["yaw", 3]]:
            Rotation = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.,
                "a1": orientation[axis-1],
                "a2": orientation[axis-1],
            }
            Translation = {
                "type": 1,
                "axis": axis,
                "t1": -0.1,
                "t2": 0.,
                "a1": -CG[axis-1],
                "a2": -CG[axis-1],
            }

            obj = Movement(name, Rotation, Translation)
            sequence.append(obj)

        for disturbance in disturbances:
            if disturbance.type is not None:
                sequence.append(distrubance2movement(disturbance))

        movement.append(sequence)
    return movement


def setParams(nBodies, nAirfoils, maxiter, timestep, Uinf, WindAngle, dens):
    params = {
        "nBods": nBodies,
        "nBlades": nAirfoils,
        "maxiter": maxiter,
        "timestep": timestep,
        "Uinf": [Uinf * np.cos(WindAngle*np.pi/180), 0.0, Uinf * np.sin(WindAngle*np.pi/180)],
        "rho": dens,
        "visc": 0.0000156,
    }
    return params


def makeSurfaceDict(surf, idx):
    s = {
        'NB': idx,
        "NACA": surf.airfoil.name,
        "name": surf.name,
        'bld': f'{surf.name}.bld',
        'cld': f'{surf.airfoil.name}.cld',
        'NNB': surf.N,
        'NCWB': surf.M,
        "x_0": surf.Origin[0] + surf.chord[0]/4,
        "y_0": surf.Origin[1],
        "z_0": surf.Origin[2],
        "pitch": surf.Orientation[0],
        "cone": surf.Orientation[1],
        "wngang": surf.Orientation[2],
        "x_end": surf.Origin[0] - surf.xoff[-1] + surf.chord[-1]/4,
        "y_end": surf.Origin[1] + surf.span,
        "z_end": surf.Origin[2] + surf.Ddihedr[-1],
        "Root_chord": surf.chord[0],
        "Tip_chord": surf.chord[-1],
        "Offset": surf.xoff[-1]
    }
    return s


def distrubance2movement(disturbance):
    if disturbance.type == "Derivative":
        t1 = -1
        t2 = 0
        a1 = 0
        a2 = disturbance.amplitude
        distType = 8
    elif disturbance.type == "Value":
        t1 = -1
        t2 = 0.
        a1 = disturbance.amplitude
        a2 = disturbance.amplitude
        distType = 1

    empty = {
        "type": 1,
        "axis": disturbance.axis,
        "t1": -1,
        "t2": 0,
        "a1": 0,
        "a2": 0,
    }

    dist = {
        "type": distType,
        "axis": disturbance.axis,
        "t1": t1,
        "t2": t2,
        "a1": a1,
        "a2": a2,
    }

    if disturbance.isRotational:
        Rotation = dist
        Translation = empty
    else:
        Rotation = empty
        Translation = dist

    return Movement(disturbance.name, Rotation, Translation)


class Movement():
    def __init__(self, name, Rotation, Translation):
        self.name = name
        self.Rtype = Rotation["type"]

        self.Raxis = Rotation["axis"]

        self.Rt1 = Rotation["t1"]
        self.Rt2 = Rotation["t2"]

        self.Ra1 = Rotation["a1"]
        self.Ra2 = Rotation["a2"]

        self.Ttype = Translation["type"]

        self.Taxis = Translation["axis"]

        self.Tt1 = Translation["t1"]
        self.Tt2 = Translation["t2"]

        self.Ta1 = Translation["a1"]
        self.Ta2 = Translation["a2"]
