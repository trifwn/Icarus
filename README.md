# ICARUS

Major Refactoring: Welcoming Version 0.2.0
---

## The project is under Work

Code and Tools to analyze the performance of low speed aircraft using different computational methods and codes. Inspired by a model aircraft designed by EUROAVIA Athens for the European Competion Air Cargo Challenge (ACC) that failed to fly because of of dynamic instability (hence the ICARUS name). This software is developed for it to be used as part of a more general optimization workflow that will use different fidelity levels to produce mission specific aircrafts.

---

## How to run

More information on how to run and compile dependencies will be availabe in the future. This project depends on the following 3d Party Software:

- OpenFoam (by OpenFoam.org)
- Foil2Wake (National Technical University of Athens Laboratory of Aerodynamics)
- XFoil (MIT)
- GenuVP (National Technical University of Athens Laboratory of Aerodynamics)

---

## Basic Modules

- Aerodynamics

Library for Aerodynamic calculations. Right now it contains a 3d lifting surface solver used to calculate aircraft aerodynamic loads

- Airfoils

This library is used to handle and generate airfoils. It is an extension of the airfoils pip module (https://pypi.org/project/airfoils/).

- Conceptual

Module for conceptual Analysis and sizing of airplanes based on constrained optimization. The final goal is that a user will be able to size an aircraft by defining mission goals and constraints

- Core

Core functions for the program. Basically any operation that is not significant enough to deserve module or doesnt clearly belong somewhere

- Database:

Storage Interface to save vehicles, airfoils, analyses, solvers and more. Currently it works with the filesystem. Uses JSON to maybe one day integrate with frontend.

- Mission

Defines the mission of the airplane and the flight envelope.

- Computation

Abstraction Layer for the solvers and running of analyses.

- Vehicle

Defines Airplane and other Vehicles. Defines an airplane as a part of wings that are themselvesade of wing segments. Each class calculates geometrical characteristics and provides io for optimization workflows

- Flight_Dynamics

Defines Flight State as a trimmed airplane position. The intnent of the class is to one day integrate the control surface movement as well and also account for transient states.

- Solvers

Integration with 3d Party Software (solvers). Handles conversion between Icarus objects and input for different solvers. Also handles the output conversion and hamdling

- Visualization

All Around visualization functions that are grouped according to their function

- Environment

Abstraction for the Environment. Usefull for calculations of fluid and thermodynamical properties at different flight envelope points

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

The end goal of the project is to come packaged with all 3d party software. Right now it is difficult fore to automate the bulding process or wrap the libraries.

---

## Tasks To-Do

- Add __init__ includes to all modules
- Add __init__ rst description to all modules
- Add doctests
- Add the control parameters
- Make Mission Class and Module the whole flight envelope
- Make low fidelity approximations. Conceptual Module more robust to work with Lagrange multipliers
- Integrate AVL
- Find alternative to matplotlib for visualization (3d graphics are slows). One alternative is plotly or julia.
- Have the grid/meshing run on julia
- Add GUI
