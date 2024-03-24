import logging
import os
import subprocess
from io import StringIO

import numpy as np

from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Database import DB
from ICARUS.Database.Database_2D import AirfoilNotFoundError, PolarsNotFoundError
from ICARUS.Environment.definition import Environment
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane
from ICARUS.Vehicle.utils import DiscretizationType
from ICARUS.Vehicle.wing_segment import Wing_Segment
from ICARUS.Airfoils.airfoil_polars import PolarNotAccurate, Polars, ReynoldsNotIncluded


def make_input_files(
    directory: str,
    plane: Airplane,
    state: State,
    solver2D: str = "Xfoil",
    solver_options: dict[str, float] = {},
) -> None:
    os.makedirs(directory, exist_ok=True)
    avl_mass(directory, plane, state.environment)
    avl_geo(directory, plane, state, state.u_freestream, solver2D, solver_options)


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
    f_io.write(
        "#  Names and scalings for units to be used for trim and eigenmode calculations.\n"
    )
    f_io.write(
        "#  The Lunit and Munit values scale the mass, xyz, and inertia table data below.\n"
    )
    f_io.write(
        "#  Lunit value will also scale all lengths and areas in the AVL input file.\n"
    )
    f_io.write("Lunit = 1 m\n")
    f_io.write("Munit = 1 kg\n")
    f_io.write("Tunit = 1.0 s\n")
    f_io.write("\n")
    f_io.write("#-------------------------\n")
    f_io.write(
        "#  Gravity and density to be used as default values in trim setup (saves runtime typing).\n"
    )
    f_io.write("#  Must be in the unit names given above (m,kg,s).\n")
    f_io.write(f"g   = {env.GRAVITY}\n")
    f_io.write(f"rho = {env.air_density}\n")
    f_io.write("\n")
    f_io.write("#-------------------------\n")
    f_io.write("#  Mass & Inertia breakdown.\n")
    f_io.write("#  x y z  is location of item's own CG.\n")
    f_io.write("#  Ixx... are item's inertias about item's own CG.\n")
    f_io.write("#\n")
    f_io.write(
        "#  x,y,z system here must be exactly the same one used in the .avl input file\n"
    )
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
    state: State,
    u_inf: float,
    solver2D: str = "Xfoil",
    solver_options: dict[str, float] = {},
) -> None:
    environment = state.environment
    if os.path.isfile(f"{PLANE_DIR}/{plane.name}.avl"):
        os.remove(f"{PLANE_DIR}/{plane.name}.avl")

    f_io = StringIO()

    # PLANE DEFINITION
    f_io.write(f"{plane.name}\n")
    f_io.write("0.0                                 | Mach\n")
    if state.ground_effect():
        h = state.environment.altitude
        f_io.write(f"0     1     {-h}                      | iYsym  iZsym  Zsym\n")
    else:
        f_io.write("0     0     0.0                      | iYsym  iZsym  Zsym\n")

    f_io.write(
        f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref\n"
    )
    f_io.write(
        f"  {0}     {0}     {0}   | Xref   Yref   Zref\n"
    )
    f_io.write(f" 0.00                               | CDp  (optional)\n")

    for i, surf in enumerate(plane.surfaces):
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("\n")

        # SURFACE DEFINITION
        f_io.write(
            f"#-------------Surface {i+1} of {len(plane.surfaces)}-----------------\n"
        )
        f_io.write("SURFACE                      | (keyword)\n")
        f_io.write(f"{surf.name}                 | surface name string \n")
        f_io.write("#Nchord    Cspace   [ Nspan Sspace ]\n")
        try:
            if surf.chord_spacing == DiscretizationType.USER_DEFINED:
                chord_spacing = DiscretizationType.COSINE.value
            else:
                chord_spacing = surf.chord_spacing.value
        except AttributeError:
            chord_spacing = DiscretizationType.COSINE.value

        if isinstance(surf, Wing_Segment):
            if surf.span_spacing == DiscretizationType.USER_DEFINED:
                span_spacing = DiscretizationType.COSINE.value
            else:
                span_spacing = surf.span_spacing.value
        else:
            span_spacing = DiscretizationType.COSINE.value

        f_io.write(f"{surf.M}        {chord_spacing} \n")
        f_io.write("\n")

        viscous = True
        if "inviscid" in solver_options.keys():
            if solver_options["inviscid"]:
                viscous = False
        # viscous = False

        f_io.write("INDEX                        | (keyword)\n")
        f_io.write(f"{int(i)}                    | SURFACE INDEX \n")

        if surf.is_symmetric_y:
            f_io.write("YDUPLICATE\n")
            f_io.write("0.0\n")
        f_io.write("SCALE\n")
        f_io.write("1.0  1.0  1.0\n")
        f_io.write("TRANSLATE\n")
        f_io.write("0.0  0.0  0.0\n")
        f_io.write("ANGLE\n")
        f_io.write(f" {surf.orientation[0]}                         | dAinc\n")
        f_io.write("\n")
        f_io.write("\n")

        for j, strip in enumerate(surf.strips):
            f_io.write(
                f"#------------ {surf.name} SECTION---{j+1} of {len(surf.strips)} of---------------------|  (keyword)\n"
            )
            f_io.write("#| Xle      Yle         Zle   Chord Ainc   [ Nspan Sspace ]\n")
            f_io.write(f"SECTION\n")
            f_io.write(
                f"   {strip.x0}    {strip.y0}    {strip.z0}    {strip.mean_chord}   {strip.mean_twist*180/np.pi}   {1}    {span_spacing}   \n",
            )
            f_io.write("\n")
            f_io.write("AFILE \n")
            f_io.write(f"{strip.mean_airfoil.file_name}\n")
            f_io.write("\n")
            f_io.write("\n")
            # Save Airfoil file
            strip.mean_airfoil.repanel_spl(180, 1e-7)
            strip.mean_airfoil.save_selig(PLANE_DIR)
            if viscous:
                # print(f"\tCalculating polar for {strip.mean_airfoil.name}")
                # Calculate average reynolds number
                reynolds = (
                    strip.mean_chord
                    * u_inf
                    / environment.air_kinematic_viscosity
                )
                # Get the airfoil polar
                try:
                    polar_obj: Polars = DB.foils_db.get_polars(
                        strip.mean_airfoil.name, solver2D
                    )
                    reyns_computed  = polar_obj.reynolds_nums

                    # print("We have computed polars for the following reynolds numbers:")
                    # print(reyns_computed)
                    # print(f"Reynolds number for this strip: {reynolds}")

                    RE_MIN = 8e4
                    RE_MAX = 1.5e6
                    NUM_BINS = 12
                    REYNOLDS_BINS = (np.logspace(-2.2, 0, NUM_BINS) * (RE_MAX - RE_MIN) + RE_MIN) 
                    DR_REYNOLDS = np.diff(REYNOLDS_BINS)

                    # Check if the reynolds number is within the range of the computed polars
                    # To be within the range of the computed polars the reynolds number must be 
                    # reyns_computed[i] - DR_REYNOLDS[matching] < reynolds_wanted < reyns_computed[i] + DR_REYNOLDS[matching]
                    # If the reynolds number is not within the range of the computed polars, the polar is recomputed
                    
                    # Find the bin corresponding to the each computed reynolds number
                    reyns_bin = np.digitize(reynolds, REYNOLDS_BINS) - 1
                    # print(REYNOLDS_BINS)
                    # print(f"Reynolds bin: {reyns_bin}")
                    cond = False
                    for i, reyns in enumerate(reyns_computed):
                        if reyns - DR_REYNOLDS[reyns_bin] < reynolds < reyns + DR_REYNOLDS[reyns_bin]:
                            cond = True
                            # print(f"\tReynolds number {reynolds} is within the range of the computed polars")
                            # print(f"   Reynolds number: {reyns} +/- {DR_REYNOLDS[reyns_bin]}")
                            break

                    if not cond:
                        DB.foils_db.compute_polars(
                            airfoil = strip.mean_airfoil,
                            solvers = [solver2D],
                            reynolds = reynolds,
                            angles = np.linspace(-10, 20, 31),
                        )
                        polar_obj: Polars = DB.foils_db.get_polars(
                            strip.mean_airfoil.name, solver2D
                        )                       

                    f_io.write("CDCL\n")
                    cl, cd = polar_obj.get_cl_cd_parabolic(reynolds)
                    f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
                    f_io.write(
                        f"{cl[0]}   {cd[0]}  {cl[1]}   {cd[1]}  {cl[2]}  {cd[2]}\n"
                    )
                    f_io.write("\n")
                except (AirfoilNotFoundError,PolarsNotFoundError, PolarNotAccurate, ReynoldsNotIncluded):
                    print(f"\tPolar for {strip.mean_airfoil.name} not found in database. Trying to recompute")
                    DB.foils_db.compute_polars(
                        airfoil = strip.mean_airfoil,
                        solvers = [solver2D],
                        reynolds = reynolds,
                        angles = np.linspace(-10, 20, 31),
                    )
                    
                    try:
                        polar_obj: Polars = DB.foils_db.get_polars(
                            strip.mean_airfoil.name, solver2D
                        )

                        f_io.write("CDCL\n")
                        cl, cd = polar_obj.get_cl_cd_parabolic(reynolds)
                        f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
                        f_io.write(
                            f"{cl[0]}   {cd[0]}  {cl[1]}   {cd[1]}  {cl[2]}  {cd[2]}\n"
                        )
                        f_io.write("\n")
                    except (PolarsNotFoundError):
                        print(f"\tCould not compute polar for {strip.mean_airfoil.name}")
                        pass                 

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
