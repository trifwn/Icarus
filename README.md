# ICARUS

Major Refactoring: Welcoming Version 0.1.0
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

-Conceptual

Module for conceptual Analysis.

-Core

Core functions for the program

-Database:

Storage Interface to save vehicles, airfoils, analyses, solvers and more. Currently it works with the filesystem. Uses JSON to maybe one day integrate with frontend.

-Mission

Defines the mission of the airplane and the flight envelope.

-Workers

Abstraction Layer for the solvers.

-Vehicle

Defines Airplane and other Vehicles.

-Flight_Dynamics

Defines Flight State 

-Software

Integration with 3d Party Software.

-Visualization

All Around visualization functions

-Environment

Abstraction for the Environment.

---

## Building and Installing the Python Module
-----------------------------------------
Currently there are some errors on building the module. Will be worked out in the future as they are not a priority and project changes rapidly. With some basic debugging it works. The most common problem is with the xfoil package.
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

- Add the control parameters
- Make Mission Class and Module the whole flight envelope
- Make low fidelity approximations. Conceptual Module
- Integrate AVL and GNVP7
- Find alternative to matplotlib for visualization (3d graphics are slows)
- Have the grid/meshing run on julia
- Add GUI
