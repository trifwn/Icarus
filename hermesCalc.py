# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from airLibs import airfoil as af
from airLibs import runF2w as f2w
from airLibs import plotting as aplt
from airLibs import runOpenFoam as of
from airLibs import runXFoil as xf


# %%
masterDir = os.getcwd()

# %% [markdown]
# # Reynolds And Mach and AoA

# %%


def ms2mach(ms):
    return ms / 340.29


def Re(v, c, n):
    return (v * c) / n


# %%
chordMax = 0.18
chordMin = 0.11
umax = 30
umin = 5
ne = 1.56e-5


# %%
Machmin = ms2mach(10)
Machmax = ms2mach(30)
Remax = Re(umax, chordMax, ne)
Remin = Re(umin, chordMin, ne)
AoAmax = 15
AoAmin = -6
NoAoA = (AoAmax - AoAmin) * 2 + 1

angles = np.linspace(AoAmin, AoAmax, NoAoA)
Reynolds = np.logspace(np.log10(Remin), np.log10(Remax), 20, base=10)
Mach = np.linspace(Machmax, Machmin, 10)

Reyn = Remin
MACH = Machmax


# %%
MACH


# %%
os.chdir(masterDir)
CASE = "Rudder"
os.chdir(CASE)
caseDir = f"Reynolds_{np.format_float_scientific(Reyn,sign=False,precision=3).replace('+', '')}"
os.system(f"mkdir -p {caseDir}")
os.chdir(caseDir)
caseDir = os.getcwd()
cleaning = False
calcF2W = False
calcOpenFoam = True
calcXFoil = False

# %% [markdown]
# # Get Airfoil

# %%
for i in os.listdir('../'):
    if i.startswith("naca"):
        airfile = i
airfoil = airfile[4:]

# %% [markdown]
# # Generate Airfoil

# %%
n_points = 100
pts = af.saveAirfoil(["s", airfile, airfoil, 0, n_points])
x, y = pts.T
plt.plot(x[: n_points], y[: n_points], "r")
plt.plot(x[n_points:], y[n_points:], "b")

# plt.plot(x,y)
plt.axis("scaled")

# %% [markdown]
# # Foil2Wake

# %%
Ncrit = 9
ftrip_low = {"pos": 0.01, "neg": 0.02}
ftrip_up = {"pos": 0.01, "neg": 0.02}

if cleaning == True:
    f2w.removeResults(angles)
if calcF2W == True:
    f2w.setupF2W()
    clcd = f2w.runFw2(Reyn, MACH, ftrip_low, ftrip_up, angles, airfile)
clcdcmFW = f2w.makeCLCD(Reyn, MACH)

# %% [markdown]
# # Xfoil

# %%
clcdcmXF = xf.runXFoil(Reyn, MACH, angles, airfoil)

# %% [markdown]
# # OpenFoam

# %%
os.chdir(caseDir)
maxITER = 10500
if cleaning == True:
    of.cleanOpenFoam()
if calcOpenFoam == True:
    of.makeMesh(airfile)
    of.setupOpenFoam(Reyn, MACH, angles, silent=True, maxITER=maxITER)
    of.runFoam(angles)
clcdcmOF = of.makeCLCD(angles)


# %%
