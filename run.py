# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import os
import time
from airLibs import setupGNVP as gnvp
# %% [markdown]
# # Parameters
AoAmax = 8
AoAmin = -8
NoAoA = (AoAmax - AoAmin) + 1
angles = np.linspace(AoAmin, AoAmax, NoAoA)

masterDir = os.getcwd()
print(masterDir)

os.chdir("3D")
os.system("rm res.dat")
os.system("rm gnvp.out")
string = "TTIME PSIB TFORC(1) TFORC(2) TFORC(3) TAMOM(1) TAMOM(2) TAMOM(3) TFORC2D(1) TFORC2D(2) TFORC2D(3) TAMOM2D(1) TAMOM2D(2) TAMOM2D(3) TFORCDS2D(1)  TFORCDS2D(2)  TFORCDS2D(3)  TAMOMDS2D(1)  TAMOMDS2D(2) TAMOMDS2D(3)"
os.system(f"echo '{string}' > res.dat")
os.chdir(masterDir)

for i, angle in enumerate(angles):
    # %%
    print(f"Running ANGLE {angle}")
    airfoils = ["4415"]
    cldata = 'MMMM'

    params = {
        "nBods": 2,  # len(Surfaces)
        "nBlades": 1,  # len(NACA)
        "maxiter": 100,
        "timestep": 0.01,
        "Uinf": [- 20 * np.sin(angle), 0.0, 20 * np.cos(angle)],
        "rho": 1.225,
        "visc": 0.0000156,
    }

    # %% [markdown]
    # # Define Airplane State

    # %%
    bodies = []
    airMovement = {
        'alpha_s': 0.,
        'alpha_e': 0.,
        'beta_s': 0.,
        'beta_e': 0.,
        'phi_s': 0.,
        'phi_e': 0.
    }

    # %% [markdown]
    # ## Left Wing

    # %%
    bodies.append({
        'NB': 1,
        "NACA": 4415,
        "name": "Lwing",
        'bld': 'Lwing.bld',
        'cld':  '4415.cld',
        'NNB': 50,
        'NCWB': 50,
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

    # %% [markdown]
    # ## Right Wing

    # %%
    bodies.append({
        'NB': 2,
        "NACA": 4415,
        "name": "Rwing",
        'bld': 'Rwing.bld',
        'cld':  '4415.cld',
        'NNB': 50,
        'NCWB': 50,
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

    # %% [markdown]
    # # Airfoil Def

    # %%

    # %% [markdown]
    # # Setup

    # %%

    # %%
    gnvp.runGNVP(airMovement, bodies, params, airfoils, '3D')
    gnvp.removeResults('3D')

    os.chdir("3D")
    os.system("./gnvp < input >> gnvp.out")
    time.sleep(0.1)
    os.system(f"cat LOADS_aer.dat >>  res.dat")
    os.chdir(masterDir)
