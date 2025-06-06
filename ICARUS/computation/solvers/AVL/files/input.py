import logging
import os
import subprocess
from io import StringIO
from typing import Any
from typing import Literal

import numpy as np
from pandas import DataFrame

from ICARUS import AVL_exe
from ICARUS.airfoils import AirfoilPolars
from ICARUS.core.types import FloatArray
from ICARUS.database import Database
from ICARUS.database import PolarsNotFoundError
from ICARUS.environment import Environment
from ICARUS.flight_dynamics import State
from ICARUS.vehicle import Airplane
from ICARUS.vehicle import DiscretizationType
from ICARUS.vehicle import WingSegment
from ICARUS.vehicle import WingSurface


def make_input_files(
    directory: str,
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
    solver_options: dict[str, Any] = {"use_avl_control": False},
) -> None:
    control_vector_dict = state.control_vector_dict
    avl_mass(directory, plane, state.environment)
    avl_geo(directory, plane, state, state.u_freestream, solver2D, solver_options)
    state.set_control(control_vector_dict)


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
        "#  Names and scalings for units to be used for trim and eigenmode calculations.\n",
    )
    f_io.write(
        "#  The Lunit and Munit values scale the mass, xyz, and inertia table data below.\n",
    )
    f_io.write(
        "#  Lunit value will also scale all lengths and areas in the AVL input file.\n",
    )
    f_io.write("Lunit = 1 m\n")
    f_io.write("Munit = 1 kg\n")
    f_io.write("Tunit = 1.0 s\n")
    f_io.write("\n")
    f_io.write("#-------------------------\n")
    f_io.write(
        "#  Gravity and density to be used as default values in trim setup (saves runtime typing).\n",
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
        "#  x,y,z system here must be exactly the same one used in the .avl input file\n",
    )
    f_io.write("#     (same orientation, same origin location, same length units)\n")
    f_io.write("#\n")
    f_io.write("#  mass   x     y     z       Ixx   Iyy   Izz    Ixy  Ixz  Iyz\n")
    f_io.write("#\n")

    for wing in plane.wings:
        f_io.write(
            f"   {wing.mass:.4e}  {wing.CG[0]:.4e}  {wing.CG[1]:.4e}  {wing.CG[2]:.4e}  {wing.Ixx:.4e}  {wing.Iyy:.4e}  {wing.Izz:.4e}  {wing.Ixy:.4e}  {wing.Ixz:.4e}  {wing.Iyz:.4e} ! {wing.name}       \n",
        )

    for mass in plane.point_masses:
        pos = mass.position
        Ixx = mass.inertia.I_xx
        Iyy = mass.inertia.I_yy
        Izz = mass.inertia.I_zz
        Ixy = mass.inertia.I_xy
        Ixz = mass.inertia.I_xz
        Iyz = mass.inertia.I_yz

        f_io.write(
            f"   {mass.mass:.4e}  {pos[0]:.4e}  {pos[1]:.4e}  {pos[2]:.4e}  {Ixx:.4e}  {Iyy:.4e}  {Izz:.4e}  {Ixy:.4e}  {Ixz:.4e}  {Iyz:.4e} ! {mass.name}     \n",
        )

    content = f_io.getvalue().expandtabs(4)
    mass_file = os.path.join(PLANE_DIR, f"{plane.name}.mass")
    with open(mass_file, "w") as massf:
        massf.write(content)


def avl_geo(
    directory: str,
    plane: Airplane,
    state: State,
    u_inf: float,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
    solver_options: dict[str, float] = {},
) -> None:
    environment = state.environment
    if os.path.isfile(f"{directory}/{plane.name}.avl"):
        os.remove(f"{directory}/{plane.name}.avl")

    f_io = StringIO()

    # PLANE DEFINITION
    f_io.write(f"{plane.name}\n")
    f_io.write("0.0                                 | Mach\n")
    if state.ground_effect():
        h = state.environment.altitude
        f_io.write(f"0     1     {-h}                      | iYsym  iZsym  Zsym\n")
    else:
        f_io.write("0     0     0                      | iYsym  iZsym  Zsym\n")

    f_io.write(
        f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref\n",
    )
    f_io.write(f"  {0}     {0}     {0}   | Xref   Yref   Zref\n")

    if "CDp" in solver_options:
        CDp = solver_options["CDp"]
    else:
        CDp = 0.0
    f_io.write(f" {CDp}                               | CDp  (optional)\n")

    surfaces: list[WingSurface] = []
    surfaces_ids = []
    for i, surface in plane.wing_segments:
        surfaces.append(surface)
        surfaces_ids.append(i)

    for i, surf in enumerate(surfaces):
        f_io.write(f"#SURFACE {i} name {surf.name}\n")

    # Use control from AVL vs ICARUS
    if "use_avl_control" in solver_options:
        state.set_control({k: 0.0 for k in state.control_vars})
        use_avl_control = True
    else:
        use_avl_control = False

    for i, surf in enumerate(surfaces):
        f_io.write("\n")
        f_io.write("\n")
        f_io.write("\n")

        # SURFACE DEFINITION
        f_io.write(
            f"#-------------Surface {i + 1} of {len(surfaces)} Surf Id {surfaces_ids[i]}-----------------\n",
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
        if "inviscid" in solver_options:
            if solver_options["inviscid"]:
                viscous = False

        if not surf.is_lifting:
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
        f_io.write(f" {surf.orientation_degrees[0]}                         | dAinc\n")
        f_io.write("\n")
        f_io.write("\n")

        for j, strip in enumerate(surf.strips):
            x, y, z = strip.leading_edge

            chord = strip.chord
            twist = strip.pitch_degrees
            strip_airfoil = strip.airfoil

            N = 1

            strip_r = np.array([x, y, z])

            f_io.write(
                f"#------------ {surf.name} SECTION---{j + 1} of {len(surf.strips)} of---------------------|  (keyword)\n",
            )
            f_io.write("#| Xle      Yle         Zle   Chord Ainc   [ Nspan Sspace ]\n")
            f_io.write("SECTION\n")
            f_io.write(
                f"   {x:.6f}    {y:.6f}    {z:.6f}    {chord:.6f}   {twist * 180 / np.pi:6f}   {N}    {span_spacing}   \n",
            )
            f_io.write("\n")
            # if strip_airfoil.file_name.upper().startswith("NACA"):
            #     f_io.write("NACA \n")
            #     f_io.write(f"{strip_airfoil.file_name[4:]}\n")
            # else:
            f_io.write("AFILE \n")
            f_io.write(f"{strip_airfoil.file_name}\n")
            f_io.write("\n")
            f_io.write("\n")

            strip_span = (surf.R_MAT.T @ strip_r)[1]
            span = surf.span
            if use_avl_control:
                for control_surf in surf.controls:
                    if (strip_span >= control_surf.span_position_start * span) and (
                        strip_span <= control_surf.span_position_end * span
                    ):
                        f_io.write("CONTROL \n")
                        f_io.write("#Cname   Cgain  Xhinge  HingeVec  SgnDup\n")
                        cname = control_surf.control_var
                        cgain = 1.0
                        if control_surf.constant_chord != 0.0:
                            x_hinge = 1 - control_surf.constant_chord / strip.chord
                        else:
                            x_hinge = control_surf.chord_function(strip_span / span)
                        # hinge_vec = surf.R_MAT.T @ control_surf.local_rotation_axis
                        sgndup = -1 if control_surf.inverse_symmetric else 1
                        f_io.write(
                            f"{cname} {-cgain}  {x_hinge} {0.0} {0.0} {0.0} {sgndup} \n",
                        )

            # Save Airfoil file
            strip_airfoil.repanel_spl(180, 1e-7)
            strip_airfoil.save_selig(directory)

            if viscous:
                # print(f"\tCalculating polar for {strip.mean_airfoil.name}")
                # Calculate average reynolds number
                reynolds = strip.chord * u_inf / environment.air_kinematic_viscosity
                # Get the airfoil polar

                try:
                    DB = Database.get_instance()
                    polar_obj: AirfoilPolars = DB.get_or_compute_airfoil_polars(
                        airfoil=strip_airfoil,
                        reynolds=reynolds,
                        solver_name=solver2D,
                        aoa=np.linspace(-10, 16, 53),
                    )
                    f_io.write("CDCL\n")
                    cl, cd, _ = polar_obj.get_cl_cd_parabolic(reynolds)
                    f_io.write("!CL1   CD1   CL2   CD2    CL3  CD3\n")
                    f_io.write(
                        f"{cl[0]:.8f}   {cd[0]:.8f}  {cl[1]:.8f}   {cd[1]:.8f}  {cl[2]:.8f}  {cd[2]:.8f}\n",
                    )
                    f_io.write("\n")
                except PolarsNotFoundError:
                    print(f"\tCould not compute polar for {strip_airfoil.name}")
            # f_io.write("CLAF\n")
            # This scales the effective dcl/da of the section airfoil as follows:
            # dcl/da  =  2 pi CLaf
            # The implementation is simply a chordwise shift of the control point
            # relative to the bound vortex on each vortex element.
            # The intent is to better represent the lift characteristics
            # of thick airfoils, which typically have greater dcl/da values
            # than thin airfoils.  A good estimate for CLaf from 2D potential
            # flow theory is
            # CLaf  =  1 + 0.77 t/c
            # where t/c is the airfoil's thickness/chord ratio.  In practice,
            # viscous effects will reduce the 0.77 factor to something less.
            # Wind tunnel airfoil data or viscous airfoil calculations should
            # be consulted before choosing a suitable CLaf value.
            # f_io.write(f"{1 + 0.77 * strip.max_thickness}\n")
            # f_io.write("\n")

    contents: str = f_io.getvalue().expandtabs(4)
    fname = f"{directory}/{plane.name}.avl"
    with open(fname, "w", encoding="utf-8") as file:
        file.write(contents)


def get_inertias(PLANEDIR: str, plane: Airplane) -> FloatArray:
    HOMEDIR = os.getcwd()
    os.chdir(PLANEDIR)

    f_io = StringIO()
    f_io.write(f"mass {plane.name}.mass\n")
    f_io.write("quit\n")
    contents: str = f_io.getvalue().expandtabs(4)

    input_fname: str = os.path.join("inertia_scr")
    log_fname = os.path.join("inertia_log.txt")

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


def get_effective_aoas(
    plane: Airplane,
    angles: FloatArray | list[float],
) -> list[DataFrame]:
    # for i, s in enumerate(plane.surfaces)
    #     if i0
    #         start += plane.surfaces[i-1].N2
    #         inds.append(start+np.arange(1,s.N+1))
    #         print(plane.surfaces[i-1].name)
    #     else
    #         inds.append(np.arange(1,s.N+1))

    dfs = []
    import pandas as pd

    from ICARUS.database import angle_to_case

    DB = Database.get_instance()
    for i, angle in enumerate(angles):
        path = os.path.join(
            DB.DB3D,
            plane.name,
            "AVL",
            f"fs_{angle_to_case(angle)}.txt",
        )
        file = open(path)
        lines = file.readlines()
        file.close()

        head: list[float] = []
        surfs: list[float] = []
        for j, k in enumerate(lines):
            if k.startswith("    j     Xle "):
                head.append(j)
            elif len(k) > 56 and (
                k[47].isdigit()
                or k[46].isdigit()
                or (k[56].isdigit() and not k.startswith("  Xref") and not k.startswith("  Sref"))
            ):
                surfs.append(j)

        surfs_arr = np.array(surfs, dtype=float)
        head_arr = np.array([head[0]], dtype=float)
        specific_rows = np.concatenate((head_arr, surfs_arr))
        df = pd.read_csv(path, sep=r"\s+", skiprows=lambda x: x not in specific_rows)
        dfs.append(df)

    return dfs
