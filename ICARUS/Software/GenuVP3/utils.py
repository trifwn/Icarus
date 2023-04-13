import numpy as np


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


def makeSurfaceDict(surf, idx, CG):
    s = {
        'NB': idx,
        "NACA": surf.airfoil.name,
        "name": surf.name,
        'bld': f'{surf.name}.bld',
        'cld': f'{surf.airfoil.name}.cld',
        'NNB': surf.M,
        'NCWB': surf.N,
        "x_0": surf.Origin[0],
        "y_0": surf.Origin[1],
        "z_0": surf.Origin[2],
        "pitch": surf.Orientation[0],
        "cone": surf.Orientation[1],
        "wngang": surf.Orientation[2],
        "x_end": surf.Origin[0] + surf.xoff[-1],
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
