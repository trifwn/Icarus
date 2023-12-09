import os
import subprocess
from enum import Enum

import numpy as np
import pandas as pd
from regex import P

from ICARUS.Computation.Solvers.AVL import Dir
from ICARUS.Computation.Solvers.AVL import DUMMY_MASS_FILE
from ICARUS.Computation.Solvers.AVL import geo_path
from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Environment.definition import Environment
from ICARUS.Vehicle.plane import Airplane


class DiscretizationType(Enum):
    """
    Discretization types for AVL
     3.0        equal         |   |   |   |   |   |   |   |   |
     2.0        sine          || |  |   |    |    |     |     |
     1.0        cosine        ||  |    |      |      |    |  ||
     0.0        equal         |   |   |   |   |   |   |   |   |
    -1.0        cosine        ||  |    |      |      |    |  ||
    -2.0       -sine          |     |     |    |    |   |  | ||
    -3.0        equal         |   |   |   |   |   |   |   |   |
    """

    EQUAL = 3.0
    COSINE = 1.0
    SINE = 2.0
    INV_SINE = -2.0
    INV_COSINE = -1.0


def make_input_files(
    PLANE_DIR: str,
    plane: Airplane,
    env: Environment,
    wing_chord_spacing: DiscretizationType,
    wing_span_spacing: DiscretizationType,
    wing_airfoil_polar: FloatArray,
    elevator_chord_spacing: DiscretizationType,
    elevator_span_spacing: DiscretizationType,
    elevator_airfoil_polar: FloatArray,
    rudder_chord_spacing: DiscretizationType,
    rudder_span_spacing: DiscretizationType,
    rudder_airfoil_polar: FloatArray,
) -> None:
    get_inertias(PLANE_DIR, plane)
    avl_mass(PLANE_DIR, plane, env)
    avl_geo(
        PLANE_DIR,
        plane,
        wing_chord_spacing,
        wing_span_spacing,
        wing_airfoil_polar,
        elevator_chord_spacing,
        elevator_span_spacing,
        elevator_airfoil_polar,
        rudder_chord_spacing,
        rudder_span_spacing,
        rudder_airfoil_polar,
    )


def avl_mass(
    PLANE_DIR: str,
    plane: Airplane,
    env: Environment,
) -> None:
    # This function creates an avl mass input file, its arguments include the masses, inertias and COGs of the various lifting and non lifting bodies-points

    with open(DUMMY_MASS_FILE) as f:
        con = f.readlines()

    ar: list[str] = [c for c in con[:30]]

    ar[1] = f"#  {plane.name} "
    ar[18] = f"rho = {env.air_density}"
    ar.append(
        f"   {plane.surfaces[0].mass}   {plane.surfaces[0].CG[0]}  {plane.surfaces[0].CG[1]}  {plane.surfaces[0].CG[2]}   {plane.surfaces[0].Ixx}   {plane.surfaces[0].Iyy}   {plane.surfaces[0].Izz} {plane.surfaces[0].Ixy}   {plane.surfaces[0].Ixz}   {plane.surfaces[0].Iyz}   ! main wing       ",
    )
    ar.append(
        f"   {plane.surfaces[2].mass}   {plane.surfaces[2].CG[0]}  {plane.surfaces[2].CG[1]}  {plane.surfaces[2].CG[2]}   {plane.surfaces[2].Ixx}   {plane.surfaces[2].Iyy}   {plane.surfaces[2].Izz} {plane.surfaces[2].Ixy}   {plane.surfaces[2].Ixz}   {plane.surfaces[2].Iyz}   ! rudder       ",
    )
    ar.append(
        f"   {plane.surfaces[1].mass}   {plane.surfaces[1].CG[0]}  {plane.surfaces[1].CG[1]}  {plane.surfaces[1].CG[2]}   {plane.surfaces[1].Ixx}   {plane.surfaces[1].Iyy}   {plane.surfaces[1].Izz} {plane.surfaces[1].Ixy}   {plane.surfaces[1].Ixz}   {plane.surfaces[1].Iyz}   ! elevator     ",
    )
    for i, m in enumerate(plane.masses[3:]):
        ar.append(
            f"   {m[0]}   {m[1][0]}  {m[1][1]}  {m[1][2]}   {0.0}   {0.0}   {0.0} {0.0}   {0.0}   {0.0}   ! {m[2]}     ",
        )

    mass_file = os.path.join(PLANE_DIR, f"{plane.name}.mass")
    with open(mass_file, "w") as massf:
        np.savetxt(massf, ar, delimiter=" ", fmt="%s")


def avl_geo(
    PLANE_DIR: str,
    plane: Airplane,
    wing_chord_spacing: DiscretizationType,
    wing_span_spacing: DiscretizationType,
    wing_airfoil_polar: FloatArray,
    elevator_chord_spacing: DiscretizationType,
    elevator_span_spacing: DiscretizationType,
    elevator_airfoil_polar: FloatArray,
    rudder_chord_spacing: DiscretizationType,
    rudder_span_spacing: DiscretizationType,
    rudder_airfoil_polar: FloatArray,
) -> None:
    if os.path.isfile(f"{PLANE_DIR}/{plane.name}.avl"):
        os.remove(f"{PLANE_DIR}/{plane.name}.avl")
    df = pd.read_csv(geo_path, on_bad_lines="skip")
    ar = df.to_numpy()
    ar[4] = f"{plane.name}"
    ar[7] = f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref"
    ar[8] = f"  {plane.CG[0]}     {plane.CG[1]}     {plane.CG[2]}   | Xref   Yref   Zref"
    ar[14] = f"{plane.surfaces[0].M}        {wing_chord_spacing}"
    ar[
        17
    ] = f"{wing_airfoil_polar[0,0]}   {wing_airfoil_polar[1,0]}  {wing_airfoil_polar[0,1]}   {wing_airfoil_polar[1,1]}  {wing_airfoil_polar[0,2]}  {wing_airfoil_polar[1,2]}"
    ar[27] = f"  {plane.surfaces[0].orientation[0]}                         | dAinc"
    ar[
        31
    ] = f"   {plane.surfaces[0].strips[0].x0}    {plane.surfaces[0].strips[0].y0}    {plane.surfaces[0].strips[0].z0}    {plane.surfaces[0].chord[0]}   {0.0}   {plane.surfaces[0].N}    {wing_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[33] = f"{Dir}/NACA{plane.surfaces[0].airfoil.name}.txt"
    ar[
        36
    ] = f"   {plane.surfaces[0].strips[-1].x1}    {plane.surfaces[0].strips[-1].y1}    {plane.surfaces[0].strips[-1].z1}    {plane.surfaces[0].chord[1]}   {0.0}   {plane.surfaces[0].N}    {wing_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[38] = f"{Dir}/NACA{plane.surfaces[0].airfoil.name}.txt"

    ar[43] = f"{plane.surfaces[1].M}        {elevator_chord_spacing}"
    ar[
        46
    ] = f"{elevator_airfoil_polar[0,0]}   {elevator_airfoil_polar[1,0]}  {elevator_airfoil_polar[0,1]}   {elevator_airfoil_polar[1,1]}  {elevator_airfoil_polar[0,2]}  {elevator_airfoil_polar[1,2]}"
    ar[
        60
    ] = f"   {plane.surfaces[1].strips[0].x0}    {plane.surfaces[1].strips[0].y0}    {plane.surfaces[1].strips[0].z0}   {plane.surfaces[1].chord[0]}   {0.0}   {plane.surfaces[1].N}    {elevator_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[62] = f"{Dir}/NACA{plane.surfaces[1].airfoil.name}.txt"
    ar[
        65
    ] = f"   {plane.surfaces[1].strips[-1].x1}    {plane.surfaces[1].strips[-1].y1}    {plane.surfaces[1].strips[-1].z1}    {plane.surfaces[1].chord[0]}   {0.0}   {plane.surfaces[1].N}    {elevator_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[67] = f"{Dir}/NACA{plane.surfaces[1].airfoil.name}.txt"
    ar[72] = f"{plane.surfaces[2].M}        {rudder_chord_spacing}"
    ar[
        75
    ] = f"{rudder_airfoil_polar[0,0]}   {rudder_airfoil_polar[1,0]}  {rudder_airfoil_polar[0,1]}   {rudder_airfoil_polar[1,1]}  {rudder_airfoil_polar[0,2]}  {rudder_airfoil_polar[1,2]}"
    ar[
        87
    ] = f"   {plane.surfaces[2].strips[0].x0}    {plane.surfaces[2].strips[0].y0}    {plane.surfaces[2].strips[0].z0}    {plane.surfaces[2].chord[0]}   {0.0}   {plane.surfaces[2].N}    {rudder_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[89] = f"{Dir}/NACA{plane.surfaces[2].airfoil.name}.txt"
    ar[
        92
    ] = f"   {plane.surfaces[2].strips[-1].x1}    {plane.surfaces[2].strips[-1].y1}    {plane.surfaces[2].strips[-1].z1}    {plane.surfaces[2].chord[0]}   {0.0}   {plane.surfaces[2].N}    {rudder_span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[94] = f"{Dir}/NACA{plane.surfaces[2].airfoil.name}.txt"

    np.savetxt(f"{PLANE_DIR}/{plane.name}.avl", ar, delimiter=" ", fmt="%s")


def get_inertias(PLANE_DIR: str, plane: Airplane) -> FloatArray:
    HOMEDIR = os.getcwd()
    os.chdir(PLANE_DIR)

    li = []
    li.append(f"mass {plane.name}.mass")
    li.append(f"quit")
    ar = np.array(li)

    input_fname: str = os.path.join(f"inertia_scr")
    log_fname = os.path.join(f"inertia_log.txt")

    np.savetxt(input_fname, ar, delimiter=" ", fmt="%s")

    with open(input_fname) as fin:
        with open(log_fname, "w") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
        print(f"AVL return code: {res}")

    with open(log_fname) as fout:
        lines = fout.readlines()

    Ixx = float(lines[38][21:28])
    Iyy = float(lines[39][33:40])
    Izz = float(lines[40][43:50])
    Ixz = float(lines[38][43:50])
    Ixy = float(lines[38][33:40])
    Iyz = float(lines[39][43:50])

    os.chdir(HOMEDIR)

    return np.array([Ixx, Iyy, Izz, Ixz, Ixy, Iyz])


# def prepro(plane):
#     os.chdir(Dir)
#     if os.path.isdir(f"{plane.name}_genD") == True:
#         # Do the same in pure python
#         files = os.listdir(f"{plane.name}_genD")
#         for f in files:
#             if f not in [
#                 f"{plane.name}.avl",
#                 f"{plane.name}.mass",
#                 f"{plane.name}_x.txt",
#                 f"{plane.name}_x.eigs",
#             ]:
#                 os.remove(f"{plane.name}_genD/{f}")
#     os.makedirs(f"{plane.name}_genD", exist_ok=True)
