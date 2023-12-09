import pandas as pd
from avl import Dir, geo_path, mass_path
import numpy as np
import os
import shutil
import pandas as pd


def avl_mass(
    plane,
    env,
    w_only,
):
    # This function creates an avl mass input file, its arguments include the masses, inertias and COGs of the various lifting and non lifting bodies-points

    pl_dir = f"{plane.name}_genD"
    if os.path.isdir(pl_dir):
        print("OPA")
        shutil.rmtree(pl_dir)
    os.mkdir(pl_dir)
    f = open(mass_path)
    con = f.readlines()
    f.close()
    ar = []
    for i, c in enumerate(con[:30]):
        ar.append(con[i])
    ar[1] = f"#  {plane.name} "
    ar[18] = f"rho = {env.air_density}"
    ar.append(
        f"   {plane.surfaces[0].mass}   {plane.surfaces[0].CG[0]}  {plane.surfaces[0].CG[1]}  {plane.surfaces[0].CG[2]}   {plane.surfaces[0].Ixx}   {plane.surfaces[0].Iyy}   {plane.surfaces[0].Izz} {plane.surfaces[0].Ixy}   {plane.surfaces[0].Ixz}   {plane.surfaces[0].Iyz}   ! main wing       "
    )
    ar.append(
        f"   {plane.surfaces[2].mass}   {plane.surfaces[2].CG[0]}  {plane.surfaces[2].CG[1]}  {plane.surfaces[2].CG[2]}   {plane.surfaces[2].Ixx}   {plane.surfaces[2].Iyy}   {plane.surfaces[2].Izz} {plane.surfaces[2].Ixy}   {plane.surfaces[2].Ixz}   {plane.surfaces[2].Iyz}   ! rudder       "
    )
    ar.append(
        f"   {plane.surfaces[1].mass}   {plane.surfaces[1].CG[0]}  {plane.surfaces[1].CG[1]}  {plane.surfaces[1].CG[2]}   {plane.surfaces[1].Ixx}   {plane.surfaces[1].Iyy}   {plane.surfaces[1].Izz} {plane.surfaces[1].Ixy}   {plane.surfaces[1].Ixz}   {plane.surfaces[1].Iyz}   ! elevator     "
    )
    for i, m in enumerate(plane.masses[3:]):
        ar.append(
            f"   {m[0]}   {m[1][0]}  {m[1][1]}  {m[1][2]}   {0.0}   {0.0}   {0.0} {0.0}   {0.0}   {0.0}   ! {m[2]}     "
        )
    # the w_only parameter makes the creation of wing-only or wing_and_horizontal_stabilizer-only aircrafts with or without point masses possible
    ar = np.array(ar)
    if w_only == "W_NM":
        ar = ar[:31]
    elif w_only == "W_E":
        ar = np.concatenate((ar[:31], ar[32:]))
    elif w_only == "W_E_NM":
        ar = np.concatenate((ar[:31], ar[32:33]))
    elif w_only == "W":
        ar = np.concatenate((ar[:31], ar[33:]))
    else:
        print("FULL MODEL")

    np.savetxt(f"./{pl_dir}/{plane.name}.mass", ar, delimiter=" ", fmt="%s")


# cREATION OF AVL GEOMETRY FILES,
# its arguments include the positions,sections,2D shapes and 2D polars,incidence angles of the various lifting surfaces
# as well as the reference coordinates and dimensions


def avl_geo(
    plane,
    w_s_chord,
    w_s_span,
    w_polar,
    el_s_chord,
    el_polar,
    rud_s_chord,
    rud_polar,
    w_only,
):
    pl_dir = f"{plane.name}_genD"
    if os.path.isdir(pl_dir):
        if os.path.isfile(f"{pl_dir}/{plane.name}.avl"):
            os.remove(f"{pl_dir}/{plane.name}.avl")
    df = pd.read_csv(geo_path, on_bad_lines="skip")
    ar = df.to_numpy()
    ar[4] = f"{plane.name}"
    ar[
        7
    ] = f"  {plane.S}     {plane.mean_aerodynamic_chord}     {plane.span}   | Sref   Cref   Bref"
    ar[
        8
    ] = f"  {plane.CG[0]}     {plane.CG[1]}     {plane.CG[2]}   | Xref   Yref   Zref"
    ar[14] = f"{plane.surfaces[0].M}        {w_s_chord}"
    ar[
        17
    ] = f"{w_polar[0,0]}   {w_polar[1,0]}  {w_polar[0,1]}   {w_polar[1,1]}  {w_polar[0,2]}  {w_polar[1,2]}"
    ar[27] = f"  {plane.surfaces[0].orientation[0]}                         | dAinc"
    ar[
        31
    ] = f"   {plane.surfaces[0].strips[0].x0}    {plane.surfaces[0].strips[0].y0}    {plane.surfaces[0].strips[0].z0}    {plane.surfaces[0].chord[0]}   {0.0}   {plane.surfaces[0].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[33] = f"{Dir}/NACA{plane.surfaces[0].airfoil.name}.txt"
    ar[
        36
    ] = f"   {plane.surfaces[0].strips[-1].x1}    {plane.surfaces[0].strips[-1].y1}    {plane.surfaces[0].strips[-1].z1}    {plane.surfaces[0].chord[1]}   {0.0}   {plane.surfaces[0].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[38] = f"{Dir}/NACA{plane.surfaces[0].airfoil.name}.txt"

    ar[43] = f"{plane.surfaces[1].M}        {el_s_chord}"
    ar[
        46
    ] = f"{el_polar[0,0]}   {el_polar[1,0]}  {el_polar[0,1]}   {el_polar[1,1]}  {el_polar[0,2]}  {el_polar[1,2]}"
    ar[
        60
    ] = f"   {plane.surfaces[1].strips[0].x0}    {plane.surfaces[1].strips[0].y0}    {plane.surfaces[1].strips[0].z0}   {plane.surfaces[1].chord[0]}   {0.0}   {plane.surfaces[1].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[62] = f"{Dir}/NACA{plane.surfaces[1].airfoil.name}.txt"
    ar[
        65
    ] = f"   {plane.surfaces[1].strips[-1].x1}    {plane.surfaces[1].strips[-1].y1}    {plane.surfaces[1].strips[-1].z1}    {plane.surfaces[1].chord[0]}   {0.0}   {plane.surfaces[1].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[67] = f"{Dir}/NACA{plane.surfaces[1].airfoil.name}.txt"
    ar[72] = f"{plane.surfaces[2].M}        {rud_s_chord}"
    ar[
        75
    ] = f"{rud_polar[0,0]}   {rud_polar[1,0]}  {rud_polar[0,1]}   {rud_polar[1,1]}  {rud_polar[0,2]}  {rud_polar[1,2]}"
    ar[
        87
    ] = f"   {plane.surfaces[2].strips[0].x0}    {plane.surfaces[2].strips[0].y0}    {plane.surfaces[2].strips[0].z0}    {plane.surfaces[2].chord[0]}   {0.0}   {plane.surfaces[2].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[89] = f"{Dir}/NACA{plane.surfaces[2].airfoil.name}.txt"
    ar[
        92
    ] = f"   {plane.surfaces[2].strips[-1].x1}    {plane.surfaces[2].strips[-1].y1}    {plane.surfaces[2].strips[-1].z1}    {plane.surfaces[2].chord[0]}   {0.0}   {plane.surfaces[2].N}    {w_s_span}   | Xle Yle Zle   Chord Ainc   [ Nspan Sspace ]"
    ar[94] = f"{Dir}/NACA{plane.surfaces[2].airfoil.name}.txt"

    # the w_only parameter makes the creation of wing-only or wing_and_horizontal_stabilizer-only aircrafts with or without point masses possible

    if w_only == "W":
        ar = ar[:39]
    elif w_only == "W_E":
        ar = ar[:68]
    else:
        print("ALL WINGS")

    np.savetxt(f"{pl_dir}/{plane.name}.avl", ar, delimiter=" ", fmt="%s")


def get_Inertias(plane):
    pl_dir = f"{plane.name}_genD"

    li = []
    li.append(f"mass {pl_dir}/{plane.name}.mass")
    li.append(f"quit")
    ar = np.array(li)
    np.savetxt(f"{pl_dir}/{plane.name}_inertia_scr", ar, delimiter=" ", fmt="%s")
    log = f"{pl_dir}/{plane.name}_inertia_log.txt"
    os.system(f"./avl.exe < {pl_dir}/{plane.name}_inertia_scr > {log}")
    f = open(log)
    lines = f.readlines()
    f.close()
    Ixx = float(lines[38][21:28])
    Iyy = float(lines[39][33:40])
    Izz = float(lines[40][43:50])
    Ixz = float(lines[38][43:50])
    Ixy = float(lines[38][33:40])
    Iyz = float(lines[39][43:50])

    return np.array([Ixx, Iyy, Izz, Ixz, Ixy, Iyz])


# def refac(plane):
#     if os.path.isfile(f"{plane}.avl"):
#         os.system(f"mkdir {plane}_genD")
#         os.system(f"mv {plane}.avl {plane}.mass {plane}_res* ./{plane}_genD")


def prepro(plane):
    os.chdir(Dir)
    if os.path.isdir(f"{plane.name}_genD") == True:
        os.system(f"cp {plane.name}_x.txt balader")
        os.system(f"cp {plane.name}_x.eigs balader_2")
        os.system(f"rm -rf {plane.name}*")
        os.system(f"cp balader {plane.name}_x.txt")
        os.system(f"cp balader_2 {plane.name}_x.eigs")
    os.system(f"mkdir {plane.name}_genD")
