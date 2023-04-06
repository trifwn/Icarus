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

## Tasks To-Do

- Relative Velocity is opposite to the freestream. Disturbance in u should be negative for positive increase
- Make visualization charts more readable by splitting into files
- Make setup.py file and requirements.txt
- Make Mission Class
- Make Load Visualization
- Make low fidelity approximations
- Create master module to manage simulations that is aware of the different software needed to generate them
- Add GUI
- Have the grid/meshing run on julia
- Find alternative to matplotlib for visualization (3d graphics are slows)