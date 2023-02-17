import numpy as np
import os
import time
from airLibs import setupGNVP as gnvp
from airLibs import runXFoil as xf


def ms2mach(ms):
    return ms / 340.29


def Re(v, c, n):
    return (v * c) / n


masterDir = os.getcwd()
print(masterDir)

CASE = '3D/'
os.chdir(CASE)
os.system("rm res.dat")
os.system("rm gnvp.out")
string = "TTIME PSIB TFORC(1) TFORC(2) TFORC(3) TAMOM(1) TAMOM(2) TAMOM(3) TFORC2D(1) TFORC2D(2) TFORC2D(3) TAMOM2D(1) TAMOM2D(2) TAMOM2D(3) TFORCDS2D(1)  TFORCDS2D(2)  TFORCDS2D(3)  TAMOMDS2D(1)  TAMOMDS2D(2) TAMOMDS2D(3)"
os.system(f"echo '{string}' > res.dat")
os.chdir(masterDir)

AoAmax = 8
AoAmin = -8
NoAoA = (AoAmax - AoAmin) + 1
angles = np.linspace(AoAmin, AoAmax, NoAoA)

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

Reyn = Remax
MACH = Machmax

airfoils = ["4415", "0008"]
cldata = []

for airfoil in airfoils:
    print(f"Getting {airfoil} 2D polars")
    Reynolds = np.logspace(np.log10(Remin), np.log10(Remax), 5, base=10)
    clcdData = []
    for Re in Reynolds:
        clcdcmXF_t = xf.runXFoil(Reyn, MACH, angles[::2], airfoil, 0.1, 0.1)
        clcdData.append(clcdcmXF_t)

    Redicts = []
    for i, batchRe in enumerate(clcdData):
        tempDict = {}
        for bathchAng in batchRe:
            tempDict[str(bathchAng[0])] = bathchAng[1:4]
        Redicts.append(tempDict)
    cldata.append(Redicts)


for i, angle in enumerate(angles):
    print(f"Running ANGLE {angle}")

    # NOT REALLY
    airMovement = {
        'alpha_s': 0.,
        'alpha_e': 0.,
        'beta_s': 0.,
        'beta_e': 0.,
        'phi_s': 0.,
        'phi_e': 0.
    }
    bodies = []

    bodies.append({
        'NB': 1,
        "NACA": 4415,
        "name": "Lwing",
        'bld': 'Lwing.bld',
        'cld':  '4415.cld',
        'NNB': 20,
        'NCWB': 20,
        'is_right': False,
        "x_0": 0.,
        "z_0": 0.,
        "y_0": 0.,
        "pitch": 2.8,
        "cone": 0.,
        "wngang": 0.,
        "x_end": 0.,
        "z_end": 0.,
        "y_end": 1.130,
        "Root_chord": 0.159,
        "Tip_chord": 0.072
    })
    bodies.append({
        'NB': 2,
        "NACA": 4415,
        "name": "Rwing",
        'bld': 'Rwing.bld',
        'cld':  '4415.cld',
        'NNB': 20,
        'NCWB': 20,
        'is_right': True,
        "x_0": 0.,
        "z_0": 0.,
        "y_0": 0.,
        "pitch": 2.8,
        "cone": 0.,
        "wngang": 0.,
        "x_end": 0.,
        "z_end": 0.,
        "y_end": 1.130,
        "Root_chord": 0.159,
        "Tip_chord": 0.072
    })

    bodies.append({
        'NB': 3,
        "NACA": '0008',
        "name": "Ltail",
        'bld': 'Ltail.bld',
        'cld':  '0008.cld',
        'NNB': 12,
        'NCWB': 12,
        'is_right': False,
        "x_0": 0.54,
        "z_0": 0.,
        "y_0": 0.,
        "pitch": 0.,
        "cone": 0.,
        "wngang": 0.,
        "x_end": 0.,
        "z_end": 0.,
        "y_end": 0.169,
        "Root_chord": 0.130,
        "Tip_chord": 0.03
    })
    bodies.append({
        'NB': 4,
        "NACA": '0008',
        "name": "Rtail",
        'bld': 'Rtail.bld',
        'cld':  '0008.cld',
        'NNB': 12,
        'NCWB': 12,
        'is_right': True,
        "x_0": 0.54,
        "z_0": 0.,
        "y_0": 0.,
        "pitch": 0.,
        "cone": 0.,
        "wngang": 0.,
        "x_end": 0.,
        "z_end": 0.,
        "y_end": 0.169,
        "Root_chord": 0.130,
        "Tip_chord": 0.03
    })

    bodies.append({
        'NB': 5,
        "NACA": '0008',
        "name": "rudder",
        'bld': 'rudder.bld',
        'cld':  '0008.cld',
        'NNB': 50,
        'NCWB': 50,
        'is_right': True,
        "x_0": 0.54,
        "z_0": 0.1,
        "y_0": 0.,
        "pitch": 0.,
        "cone": 0.,
        "wngang": 90.,
        "x_end": 0.,
        "z_end": 0.,
        "y_end": 0.169,
        "Root_chord": 0.130,
        "Tip_chord": 0.03
    })

    params = {
        "nBods": len(bodies),  # len(Surfaces)
        "nBlades": len(airfoils),  # len(NACA)
        "maxiter": 100,
        "timestep": 0.01,
        "Uinf": [20. * np.cos(angle*np.pi/180), 0.0, 20. * np.sin(angle*np.pi/180)],
        "rho": 1.225,
        "visc": 0.0000156,
    }
    print(
        f"Velocity is {[20. * np.cos(angle*np.pi/180), 0.0, 20. * np.sin(angle*np.pi/180)]}")
    gnvp.runGNVP(airMovement, bodies, params, airfoils,
                 cldata, Reynolds, angles, CASE)
    gnvp.removeResults(CASE)

    os.chdir(CASE)
    os.system("./gnvp < input >> gnvp.out")
    time.sleep(0.1)
    os.system(f"cat LOADS_aer.dat >>  res.dat")
    os.chdir(masterDir)
