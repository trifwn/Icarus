import logging
import os
import subprocess
from io import StringIO

import numpy as np
from pandas import DataFrame

from ICARUS.airfoils.airfoil_polars import PolarNotAccurate
from ICARUS.airfoils.airfoil_polars import Polars
from ICARUS.airfoils.airfoil_polars import ReynoldsNotIncluded
from ICARUS.core.types import FloatArray
from ICARUS.database import AVL_exe
from ICARUS.database import DB
from ICARUS.database.database2D import AirfoilNotFoundError
from ICARUS.database.database2D import PolarsNotFoundError
from ICARUS.environment.definition import Environment
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.merged_wing import MergedWing
from ICARUS.vehicle.plane import Airplane
from ICARUS.vehicle.utils import DiscretizationType
from ICARUS.vehicle.wing_segment import WingSegment


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

    f_io.write(f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref\n")
    f_io.write(f"  {0}     {0}     {0}   | Xref   Yref   Zref\n")
    f_io.write(f" 0.0010                               | CDp  (optional)\n")

    surfaces = []
    surfaces_ids = []
    i = 0
    for surface in plane.surfaces:
        i += 1
        if isinstance(surface, MergedWing):
            for sub_surface in surface.wing_segments:
                surfaces.append(sub_surface)
                surfaces_ids.append(i)
        else:
            surfaces.append(surface)
            surfaces_ids.append(i)
    for i, surf in enumerate(surfaces):
        f_io.write(f"#SURFACE {i} name {surf.name}\n")

    for i, surf in enumerate(surfaces):
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("\n")

        # SURFACE DEFINITION
        f_io.write(f"#-------------Surface {i+1} of {len(surfaces)} Surf Id {surfaces_ids[i]}-----------------\n")
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

        if isinstance(surf, WingSegment):
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

        if surf.is_lifting == False:
            viscous = False
            f_io.write("\n")
            f_io.write("NOWAKE\n")
            f_io.write("\n")

        f_io.write("INDEX                        | (keyword)\n")
        f_io.write(f"{int(surfaces_ids[i])}                    | SURFACE INDEX \n")

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
        # cntrl_index = 1
        for j, strip in enumerate(surf.strips):
            f_io.write(
                f"#------------ {surf.name} SECTION---{j+1} of {len(surf.strips)} of---------------------|  (keyword)\n",
            )
            f_io.write("#| Xle      Yle         Zle   Chord Ainc   [ Nspan Sspace ]\n")
            f_io.write(f"SECTION\n")

            if j == 0:
                f_io.write(
                    f"   {strip.x0}    {strip.y0}    {strip.z0}    {strip.chords[0]}   {strip.twists[0]*180/np.pi}   {1}    {span_spacing}   \n",
                )
                f_io.write("\n")
                f_io.write("AFILE \n")
                f_io.write(f"{strip.airfoil_start.file_name}\n")
                f_io.write("\n")
                f_io.write("\n")
                strip_airfoil = strip.airfoil_start
            elif j == len(surf.strips) - 1:
                f_io.write(
                    f"   {strip.x1}    {strip.y1}    {strip.z1}    {strip.chords[1]}   {strip.twists[1]*180/np.pi}   {1}    {span_spacing}   \n",
                )
                f_io.write("\n")
                f_io.write("AFILE \n")
                f_io.write(f"{strip.airfoil_end.file_name}\n")
                f_io.write("\n")
                f_io.write("\n")
                strip_airfoil = strip.airfoil_end
            else:
                f_io.write(
                    f"   {(strip.x0 + strip.x1)/2}    {(strip.y0+strip.y1)/2}    {(strip.z0+strip.z1)/2}    {strip.mean_chord}   {strip.mean_twist*180/np.pi}   {1}    {span_spacing}   \n",
                )
                f_io.write("\n")
                f_io.write("AFILE \n")
                f_io.write(f"{strip.mean_airfoil.file_name}\n")
                f_io.write("\n")
                f_io.write("\n")
                strip_airfoil = strip.mean_airfoil

                strip_r = np.array([strip.x1, strip.y1, strip.z1])
                strip_span = (surf.R_MAT.T @ strip_r)[1]

                for control_surf in surf.controls:
                    if (strip_span >= control_surf.span_position_start) and (
                        strip_span <= control_surf.span_position_end
                    ):

                        f_io.write("CONTROL \n")
                        f_io.write("#Cname   Cgain  Xhinge  HingeVec  SgnDup\n")
                        cname = control_surf.control_var
                        cgain = 1.0
                        x_hinge = control_surf.chord_function(0.0)
                        hinge_vec = surf.R_MAT.T @ control_surf.local_rotation_axis
                        sgndup = -1 if control_surf.inverse_symmetric else 1
                        f_io.write(
                            f"{cname} {cgain}  {x_hinge} {hinge_vec[0]} {hinge_vec[1]} {hinge_vec[2]} {sgndup} \n",
                        )

            # Save Airfoil file
            strip_airfoil.repanel_spl(180, 1e-7)
            strip_airfoil.save_selig(PLANE_DIR)

            if viscous:
                # print(f"\tCalculating polar for {strip.mean_airfoil.name}")
                # Calculate average reynolds number
                reynolds = strip.mean_chord * u_inf / environment.air_kinematic_viscosity
                # Get the airfoil polar
                try:
                    polar_obj: Polars = DB.foils_db.get_polars(strip_airfoil.name, solver=solver2D)
                    reyns_computed = polar_obj.reynolds_nums

                    # print("We have computed polars for the following reynolds numbers:")
                    # print(reyns_computed)
                    # print(f"Reynolds number for this strip: {reynolds}")

                    RE_MIN = 8e4
                    RE_MAX = 1.5e6
                    NUM_BINS = 12
                    REYNOLDS_BINS = np.logspace(-2.2, 0, NUM_BINS) * (RE_MAX - RE_MIN) + RE_MIN
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
                            airfoil=strip_airfoil,
                            solvers=[solver2D],
                            reynolds=reynolds,
                            angles=np.linspace(-8, 20, 29),
                        )
                        polar_obj = DB.foils_db.get_polars(strip_airfoil.name, solver=solver2D)

                    f_io.write("CDCL\n")
                    cl, cd = polar_obj.get_cl_cd_parabolic(reynolds)
                    f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
                    f_io.write(f"{cl[0]}   {cd[0]}  {cl[1]}   {cd[1]}  {cl[2]}  {cd[2]}\n")
                    f_io.write("\n")
                except (
                    AirfoilNotFoundError,
                    PolarsNotFoundError,
                    PolarNotAccurate,
                    ReynoldsNotIncluded,
                    FileNotFoundError,
                ):
                    print(f"\tPolar for {strip_airfoil.name} not found in database. Trying to recompute")
                    DB.foils_db.compute_polars(
                        airfoil=strip_airfoil,
                        solvers=[solver2D],
                        reynolds=reynolds,
                        angles=np.linspace(-10, 20, 31),
                    )

                    try:
                        polar_obj = DB.foils_db.get_polars(strip_airfoil.name, solver=solver2D)

                        f_io.write("CDCL\n")
                        cl, cd = polar_obj.get_cl_cd_parabolic(reynolds)
                        f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
                        f_io.write(f"{cl[0]}   {cd[0]}  {cl[1]}   {cd[1]}  {cl[2]}  {cd[2]}\n")
                        f_io.write("\n")
                    except PolarsNotFoundError:
                        print(f"\tCould not compute polar for {strip_airfoil.name}")
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


def get_effective_aoas(plane: Airplane, angles: FloatArray | list[float]) -> list[DataFrame]:
    # for i, s in enumerate(plane.surfaces)
    #     if i0
    #         start += plane.surfaces[i-1].N2
    #         inds.append(start+np.arange(1,s.N+1))
    #         print(plane.surfaces[i-1].name)
    #     else
    #         inds.append(np.arange(1,s.N+1))

    dfs = []
    from ICARUS.database.utils import angle_to_case

    import pandas as pd

    for i, angle in enumerate(angles):
        path = os.path.join(DB.vehicles_db.DATADIR, plane.name, "AVL", f"fs_{angle_to_case(angle)}.txt")
        file = open(path)
        lines = file.readlines()
        file.close()

        head: list[float] = []
        surfs: list[float] = []
        for j, l in enumerate(lines):
            if l.startswith(f"    j     Xle "):
                head.append(j)
            elif len(l) > 56 and (
                l[47].isdigit()
                or l[46].isdigit()
                or l[56].isdigit()
                and not l.startswith("  Xref")
                and not l.startswith("  Sref")
            ):
                surfs.append(j)

        surfs_arr = np.array(surfs, dtype=float)
        head_arr = np.array([head[0]], dtype=float)
        specific_rows = np.concatenate((head_arr, surfs_arr))
        df = pd.read_csv(path, delim_whitespace=True, skiprows=lambda x: x not in specific_rows)
        dfs.append(df)

    return dfs
