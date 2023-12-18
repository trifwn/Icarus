import logging
import os
import subprocess
from io import StringIO
from re import S

import numpy as np
from pandas import DataFrame

from ICARUS.Airfoils.airfoil_polars import Polars
from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Database import DB
from ICARUS.Database import DB3D
from ICARUS.Environment.definition import Environment
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import DiscretizationType


def make_input_files(
    directory: str,
    plane: Airplane,
    state: State,
    solver2D: str = "Xfoil",
) -> None:
    os.makedirs(directory, exist_ok=True)
    avl_mass(directory, plane, state.environment)
    avl_geo(directory, plane, state.environment, state.u_freestream, solver2D)


def avl_mass(
    PLANE_DIR: str,
    plane: Airplane,
    env: Environment,
) -> None:
    # This function creates an avl mass input file, its arguments include the masses, inertias and COGs of the various lifting and non lifting bodies-points

    f_io = StringIO()

    f_io.write("#-------------------------------------------------\n")
    f_io.write(f"#  {plane.name}    n")
    f_io.write("#\n")
    f_io.write("#  Dimensional unit and parameter data.\n")
    f_io.write("#  Mass & Inertia breakdown.\n")
    f_io.write("#-------------------------------------------------\n")
    f_io.write("\n")
    f_io.write("#  Names and scalings for units to be used for trim and eigenmode calculations.\n")
    f_io.write("#  The Lunit and Munit values scale the mass, xyz, and inertia table data below.\n")
    f_io.write("#  Lunit value will also scale all lengths and areas in the AVL input file.\n")
    f_io.write("Lunit = 1 m\n")
    f_io.write("Munit = 1 kg\n")
    f_io.write("Tunit = 1.0 s\n")
    f_io.write("\n")
    f_io.write("#-------------------------\n")
    f_io.write("#  Gravity and density to be used as default values in trim setup (saves runtime typing).\n")
    f_io.write("#  Must be in the unit names given above (m,kg,s).\n")
    f_io.write(f"g   = {env.GRAVITY}\n")
    f_io.write(f"rho = {env.air_density}\n")
    f_io.write("\n")
    f_io.write("#-------------------------\n")
    f_io.write("#  Mass & Inertia breakdown.\n")
    f_io.write("#  x y z  is location of item's own CG.\n")
    f_io.write("#  Ixx... are item's inertias about item's own CG.\n")
    f_io.write("#\n")
    f_io.write("#  x,y,z system here must be exactly the same one used in the .avl input file\n")
    f_io.write("#     (same orientation, same origin location, same length units)\n")
    f_io.write("#\n")
    f_io.write("#  mass   x     y     z       Ixx   Iyy   Izz    Ixy  Ixz  Iyz\n")
    f_io.write("#\n")

    for surf in plane.surfaces:
        f_io.write(
            f"   {surf.mass}   {surf.CG[0]}  {surf.CG[1]}  {surf.CG[2]}   {surf.Ixx}   {surf.Iyy}   {surf.Izz} {surf.Ixy}   {surf.Ixz}   {surf.Iyz}   ! {surf.name}       \n",
        )

    for i, m in enumerate(plane.masses[len(plane.surfaces) :]):
        f_io.write(
            f"   {m[0]}   {m[1][0]}  {m[1][1]}  {m[1][2]}   {0.0}   {0.0}   {0.0} {0.0}   {0.0}   {0.0}   ! {m[2]}     \n",
        )

    content = f_io.getvalue().expandtabs(4)
    mass_file = os.path.join(PLANE_DIR, f"{plane.name}.mass")
    with open(mass_file, "w") as massf:
        massf.write(content)


def avl_geo(
    PLANE_DIR: str,
    plane: Airplane,
    environment: Environment,
    u_inf: float,
    solver2D: str = "Xfoil",
) -> None:
    if os.path.isfile(f"{PLANE_DIR}/{plane.name}.avl"):
        os.remove(f"{PLANE_DIR}/{plane.name}.avl")

    f_io = StringIO()
    f_io.write("# Note : check consistency of area unit and length units in this file\n")
    f_io.write("# Note : check consistency with inertia units of the .mass file\n")
    f_io.write("#\n")
    f_io.write("#\n")
    f_io.write(f"{plane.name}\n")
    f_io.write("0.0                                 | Mach\n")
    # !TODO : ADD SYMMETRY
    f_io.write("0     0     0.0                      | iYsym  iZsym  Zsym\n")
    f_io.write(f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref\n")
    f_io.write(f"  {plane.CG[0]}     {plane.CG[1]}     {plane.CG[2]}   | Xref   Yref   Zref\n")
    f_io.write(f" 0.00                               | CDp  (optional)\n")

    for i, surf in enumerate(plane.surfaces):
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("#========TODO: REMOVE OR MODIFY MANUALLY DUPLICATE SECTIONS IN SURFACE DEFINITION=========\n")
        f_io.write("SURFACE                      | (keyword)\n")
        f_io.write(f"{surf.name}\n")
        f_io.write("#Nchord    Cspace   [ Nspan Sspace ]\n")
        if surf.chord_spacing == DiscretizationType.UNKNOWN:
            chord_spacing = DiscretizationType.COSINE.value
        else:
            chord_spacing = surf.chord_spacing.value
        f_io.write(f"{surf.M}        {chord_spacing}\n")
        f_io.write("\n")
        f_io.write("CDCL\n")

        # Get the airfoil polar
        foil_dat = DB.foils_db.data
        try:
            polars: dict[str, DataFrame] = foil_dat[surf.root_airfoil.name][solver2D]
        except KeyError:
            try:
                polars = foil_dat[f"NACA{surf.root_airfoil.name}"][solver2D]
            except KeyError:
                raise KeyError(f"Airfoil {surf.root_airfoil.name} not found in database")

        polar_obj = Polars(polars)
        # Calculate average reynolds number
        reynolds = surf.mean_aerodynamic_chord * u_inf / environment.air_dynamic_viscosity

        cl, cd = polar_obj.get_cl_cd_parabolic(reynolds)
        f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
        f_io.write(f"{cl[0]}   {cd[0]}  {cl[1]}   {cd[1]}  {cl[2]}  {cd[2]}\n")
        f_io.write("\n")

        f_io.write("INDEX                        | (keyword)\n")
        f_io.write(f"{int(i+ 6679)}                         | Lsurf\n")

        if surf.is_symmetric_y:
            f_io.write("YDUPLICATE\n")
            f_io.write("0.0\n")
        f_io.write("SCALE\n")
        f_io.write("1.0  1.0  1.0\n")
        f_io.write("TRANSLATE\n")
        f_io.write("0.0  0.0  0.0\n")
        f_io.write("ANGLE\n")
        f_io.write(f"  {surf.orientation[0]}                         | dAinc\n")
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("#____PANEL 1_______\n")
        f_io.write("#______________\n")
        f_io.write("SECTION                                                     |  (keyword)\n")

        if surf.span_spacing == DiscretizationType.UNKNOWN:
            span_spacing = DiscretizationType.COSINE.value
        else:
            span_spacing = surf.span_spacing.value

        f_io.write(
            f"   {surf.strips[0].x0}    {surf.strips[0].y0}    {surf.strips[0].z0}    {surf.chord[0]}   {0.0}   {surf.N}    {span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]\n",
        )
        f_io.write("\n")
        f_io.write("AFIL 0.0 1.0\n")
        f_io.write(f"{surf.root_airfoil.file_name}\n")
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("#______________\n")
        f_io.write("SECTION                                                    |  (keyword)\n")
        f_io.write(
            f"{surf.strips[-1].x1}    {surf.strips[-1].y1}    {surf.strips[-1].z1}  {surf.chord[-1]}   {0.0}   {surf.N}    {span_spacing}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]\n",
        )
        f_io.write("AFIL\n")

        f_io.write(f"{surf.root_airfoil.file_name}\n")
        f_io.write("\n")

        # Save Airfoil file
        surf.root_airfoil.save_selig_te(PLANE_DIR, header=True, inverse=True)

    contents: str = f_io.getvalue().expandtabs(4)
    fname = f"{PLANE_DIR}/{plane.name}.avl"
    with open(fname, "w", encoding="utf-8") as file:
        file.write(contents)


def get_inertias(PLANEDIR: str, plane: Airplane) -> FloatArray:
    HOMEDIR = os.getcwd()
    os.chdir(PLANEDIR)

    f_io = StringIO()
    f_io.write(f"mass {plane.name}.mass\n")
    f_io.write(f"quit\n")
    contents: str = f_io.getvalue().expandtabs(4)

    input_fname: str = os.path.join(f"inertia_scr")
    log_fname = os.path.join(f"inertia_log.txt")

    with open(input_fname, "w", encoding="utf-8") as f:
        f.writelines(contents)

    with open(input_fname) as fin:
        with open(log_fname, "w") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
        logging.debug(f"AVL return code: {res}")

    with open(log_fname) as fout:
        lines = fout.readlines()

    Ixx = float(lines[38][21:28])
    Iyy = float(lines[39][32:43])
    Izz = float(lines[40][44:55])
    Ixz = float(lines[38][44:55])
    Ixy = float(lines[38][32:43])
    Iyz = float(lines[39][44:55])

    os.chdir(HOMEDIR)

    return np.array([Ixx, Iyy, Izz, Ixz, Ixy, Iyz])
