# ICARUS
Version v1.0
<p align="center">
  <img src="https://github.com/user-attachments/assets/5bbbd72c-0046-4b50-85b0-facba5348f80" alt="Vortex Wake">
</p>

## Working Effort.
Code and Tools to analyze the performance of low speed aircraft using different computational methods and codes. Inspired by a model aircraft designed by EUROAVIA Athens for the European Competion Air Cargo Challenge (ACC) that failed to fly because of of dynamic instability (hence the ICARUS name). This software is developed for it to be used as part of a more general optimization workflow that will use different fidelity levels to produce mission specific aircrafts.

## Current Goals

- Improve Examples
- Create functional API (instead of object oriented on) for analysis
- Convert to jax typing
- Finish integrating Control Surfaces for AVL, GenuVP
- Ameliorate Trajectory Calculations
- Create Panel/Particle Solver (insted of LSPT) to deprecate external aerodynamic solvers.
- Create Results Class to handle Analysis output.
- Improve Airfoil Polar class to handle higher order jax compatible interpolations
- Create Airplane Polar class
- Create Workflow Class to combine Analyses Together

## Long Term Goals
- Add more optimization techniques (stohastic, GAs, swarm, etc.)
- Integrate MDAO capabilities fully
- Integrate DATCOM for conceputal design. Also add Lower Order Models and Analytical Formulas for Stability
- Integrate Different fidelity levels for improved optimization
- Improve testing
- Finish Documentation and add doctests
- Add CD/CI
- Create fully working demo of design an airplane from start to finish in ICARUS
- Add GUI and find alternative to matplotlib for visualization (3d graphics are slows). One alternative is plotly or julia

---



---

## Basic Modules

- Aerodynamics

Library for Aerodynamic calculations. Right now it contains a 3d lifting surface solver used to calculate aircraft aerodynamic loads

- Airfoils

This library is used to handle and generate airfoils. It is an extension of the airfoils pip module (https://pypi.org/project/airfoils/).

- Computation

Abstraction Layer for the solvers and running of analyses. Handles the translation of internal objects to solver specific formats. It contains the solvers and analyses

- Conceptual

Module for conceptual Analysis and sizing of airplanes based on constrained optimization. The final goal is that a user will be able to size an aircraft by defining mission goals and constraints

- Core

Core functions for the program. Basically any operation that is not significant enough to deserve module or doesnt clearly belong somewhere

- Database:

Storage Interface to save vehicles, airfoils, analyses, solvers and more. Currently it works with the filesystem. Uses JSON to maybe one day integrate with frontend.

- Environment

Abstraction for the Environment. Usefull for calculations of fluid and thermodynamical properties at different flight envelope points

- Flight_Dynamics

Defines Flight State as a trimmed airplane position. The intnent of the class is to one day integrate the control surface movement as well and also account for transient states.

- Mission

Defines the mission of the airplane and the flight envelope.

- Optimization

Defines Optimizers for Different Solvers, Integrators and other functions that are used to optimize aerodynamic performance

- Vehicle

Defines Airplane and other vehicles. Defines an airplane as a part of wings that are themselvesade of wing segments. Each class calculates geometrical characteristics and provides io for optimization workflows


- Visualization

All Around visualization functions that are grouped according to their function


---

## Building and Installing the Python Module
-----------------------------------------
```
python -m venv venv
.\venv\Scripts\activate # Windows
./venv/bin/activate # Linux
```

To install the package run:
```bash
pip install .
```
---

## External Dependencies
The end goal of the project is to come packaged with all 3d party software. Right now it is difficult fore to automate the bulding process or wrap the libraries.
More information on how to run and compile dependencies will be availabe in the future. This project depends on the following 3d Party Software for aerodynamic calculations:

- OpenFoam (by OpenFoam.org)
- Foil2Wake (National Technical University of Athens Laboratory of Aerodynamics)
- XFoil (MIT)
- GenuVP (National Technical University of Athens Laboratory of Aerodynamics)

---
