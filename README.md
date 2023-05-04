# ICARUS

---

## The project is under Work

Code and Tools to analyze the performence of low speed aircraft using different computational methods and codes. Inspired by a model aircraft designed by EUROAVIA Athens for the European Competion Air Cargo Challenge (ACC) that failed to fly because of of dynamic instability (hence the ICARUS name). This software is developed for it to be used as part of a more general optimization workflow that will use different fidelity levels to produce mission specific aircrafts.

---

## How to run

More information on how to run and compile dependencies will be availabe in the future. This project depends on the following 3d Party Software:

- OpenFoam (by OpenFoam.org)
- Foil2Wake (National Technical University of Athens Laboratory of Aerodynamics)
- XFoil (MIT)
- GenuVP (National Technical University of Athens Laboratory of Aerodynamics)

---

## Basic Modules

-Airfoils

This library is used to handle and generate airfoils. It is an extension of the airfoils pip module (https://pypi.org/project/airfoils/).

-Core

Core functions for the program

-Mission

Defines the mission of the airplane and the flight envelope.

-Solver

Abstraction Layer for the solvers

-Plane Definition

Defines Airplane

-Flight_Dynamics

Defines Dynamic Airplane

-Software

Integration with 3d Party Software.

-Visualization

All Around visualization functions

-Control

Early Stages not much there.

---

## Building and Installing the Python Module
-----------------------------------------
If you want you can create a venv enviroment first:
```
python -m venv venv
.\venv\Scripts\activate # Windows
./venv/bin/activate # Linux
```

To install the package run:
```bash
pip install .
```

On Windows to install the xfoil package, you may have to force the system to use MinGW.
If the installation fails change the comments on  `pyproject.toml` at the root of the repo to:

```
    "xfoil @git+https://github.com/trifwn/xfoil-python-windows.git",
    #"xfoil @git+https://github.com/DARcorporation/xfoil-python.git",
```

---

## Tasks To-Do

- MAKE DATABASE USE STRUCT DATA
- Make requirements.txt
- Make Mission Class
- Make Load Visualization
- Make low fidelity approximations
- Create master module to manage simulations that is aware of the different software needed to generate them
- Find alternative to matplotlib for visualization (3d graphics are slows)
- Have the grid/meshing run on julia
- Add GUI
