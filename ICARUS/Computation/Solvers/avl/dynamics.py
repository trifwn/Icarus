import os
import shutil
import subprocess
from genericpath import exists
from math import e

import numpy as np
from numpy import deg2rad
from numpy import floating
from numpy import rad2deg

from ICARUS.Computation.Solvers.AVL import Dir
from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Vehicle.plane import Airplane


def split_file(input_file, output_prefix, delimiter):
    with open(input_file) as infile:
        data = infile.read()

    parts = data.split(delimiter)

    for i, part in enumerate(parts, start=1):
        output_file = f"{output_prefix}_{i}.txt"
        with open(output_file, "w") as outfile:
            outfile.write(part)


# EIGENVALUE ANALYSIS BASED ON THE IMPLICIT DIFFERENTIATION APPROACH OF M.DRELA AND AVL
def implicit_eigs(plane: Airplane):
    PLANE_DIR = f"{plane.name}_genD"
    CASEDIR = f"{plane.name}_polarD"
    DYNAMICS_DIR = f"{plane.name}_eigD"
    DIRNOW: str = os.getcwd()

    log = f"{DYNAMICS_DIR}/{plane.name}_eiglog.txt"
    os.makedirs(DYNAMICS_DIR, exist_ok=True)

    f_li = []
    f_li.append(f"load {PLANE_DIR}/{plane.name}.avl")
    f_li.append(f"mass {PLANE_DIR}/{plane.name}.mass")
    f_li.append("MSET")
    f_li.append("0")
    f_li.append("oper")
    f_li.append("1")
    f_li.append("a")
    f_li.append("pm")
    f_li.append("0")
    f_li.append("x")
    f_li.append("C1")
    f_li.append("b")
    f_li.append("0")
    f_li.append("    ")
    f_li.append("x")
    f_li.append("    ")
    f_li.append("mode")
    f_li.append("1")
    f_li.append("N")
    f_li.append("    ")
    f_li.append("quit")
    ar = np.array(f_li)
    np.savetxt(f"{DYNAMICS_DIR}/{plane.name}_mode_scr", ar, delimiter=" ", fmt="%s")

    input_f = os.path.join(DYNAMICS_DIR, f"{plane.name}_mode_scr")
    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [os.path.join(DYNAMICS_DIR, "avl")],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )

    char = "'{*}'"
    split_file(log, "xx", char)
    p = "xx10"
    with open(p, "w", encoding="utf-8") as f:
        lines = f.readlines()

    long_li = []
    lat_li = []
    ind_li = np.arange(0, 8, 1) * 6
    for i in ind_li:
        vals = eigpro(i, lines)
        if np.mean(np.abs(vals[0])) == 0:
            lat_li.append(vals[2])
        elif np.mean(np.abs(vals[1])) == 0:
            long_li.append(vals[2])

    # Remove the temporary files
    # Get all the files in the directory
    # Filter all files that contain the string 'xx'
    files: list[str] = os.listdir(DIRNOW)
    for file in files:
        if file.startswith("xx"):
            os.remove(file)

    return long_li, lat_li


# NECESSARY FUNCTION FOR THE ABOVE
def eigpro(st, lines):
    st = int(st)
    mode = np.array([float(lines[st][10:22]), float(lines[st][24:32])])
    long_vecs = np.zeros((4, 2), dtype=floating)
    lat_vecs = np.zeros((4, 2), dtype=floating)
    for i in range(0, 4):
        long_vecs[i, :] = np.array([float(lines[st + i + 1][9:15]), float(lines[st + i + 1][20:26])])
        lat_vecs[i, :] = np.array([float(lines[st + i + 1][41:47]), float(lines[st + i + 1][52:58])])

    return long_vecs, lat_vecs, mode


# FUCNTION OBTAINING EIGENVALUES FROM PRE-RUN XFLR5 DYNAMIC ANALYSIS (FOR COMPARISON)
def xflr_eigs(plane):
    p = f"{Dir}/{plane.name}_x.eigs"
    f = open(p)
    lines = f.readlines()
    long_li = lines[105]
    lat_li = lines[116]
    eig_mat = np.zeros((5, 2))
    eig_mat[0, 0] = float(long_li[22:38].partition("+")[0])
    eig_mat[0, 1] = float(long_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[1, 0] = float(long_li[74:92].partition("+")[0])
    eig_mat[1, 1] = float(long_li[74:92].partition("+")[2].partition("i")[0])
    eig_mat[2, 0] = float(lat_li[22:38].partition("+")[0])
    eig_mat[2, 1] = float(lat_li[22:38].partition("+")[2].partition("i")[0])
    eig_mat[3, 0] = float(lat_li[46:67].partition("+")[0])
    eig_mat[3, 1] = float(lat_li[46:67].partition("+")[2].partition("i")[0])
    eig_mat[4, 0] = float(lat_li[99:119].partition("+")[0])
    eig_mat[4, 1] = float(lat_li[99:119].partition("+")[2].partition("i")[0])

    f.close()
    print("1) Short - Period, 2) Phygoid, 3) Roll-Damping, 4) Dutch Roll, 5) Spiral")
    return eig_mat


# CALCULATION OF AIRCRAFT TRIM AOA AND VELOCITY
def trim_conditions(plane):
    trim_path = f"{plane.name}_trimD"
    pl_dir = f"{plane.name}_genD"
    if os.path.isdir(f"{plane.name}_trimD"):
        os.rmdir(trim_path)
    os.makedirs(trim_path)
    li = []

    li.append(f"load {pl_dir}/{plane.name}.avl")
    li.append(f"mass {pl_dir}/{plane.name}.mass")
    li.append(f"MSET")
    li.append("0")
    li.append(f"oper")
    li.append("1")
    li.append(f"a")
    li.append("pm")
    li.append("0")
    li.append("x")
    li.append("c1")
    li.append("b")
    li.append("0")
    li.append("    ")
    li.append("x")
    li.append("FT")
    li.append(f"{trim_path}/{plane.name}_trimmed_res.txt")
    if os.path.isfile(f"{trim_path}/{plane.name}_trimmed_res.txt"):
        li.append("O")
    li.append("    ")
    li.append("quit")
    ar = np.array(li)
    np.savetxt(f"{trim_path}/{plane.name}_trimmed_scr", ar, delimiter=" ", fmt="%s")
    log = f"{trim_path}/{plane.name}_trim_log.txt"

    input_f = os.path.join(trim_path, f"{plane.name}_trimmed_scr")
    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )

    char = "'{*}'"
    log = "your_log_file.txt"  # Replace with the actual log file path

    # Split the log file using Python
    with open(log) as infile:
        data = infile.read()

    parts = data.split("Setup")

    for i, part in enumerate(parts, start=2):  # Start from 2 because the first part is empty
        output_file = f"xx{i:02d}"
        with open(output_file, "w") as outfile:
            outfile.write(part)

    # Read lines from xx02
    p = "xx02"
    with open(p) as f:
        lines = f.readlines()

    # Remove files starting with "xx"
    for filename in os.listdir():
        if filename.startswith("xx"):
            os.remove(filename)

    V = float(lines[5][22:28])
    aoa = float(lines[68][10:19])
    return aoa, V


# FUNCTION FOR THE CALCULATION OF STABILITY DERIVATIVES VIA FINITED DIFEERENCE METHOD
def finite_difs(
    plane: Airplane,
    u_ar: FloatArray,
    v_ar: FloatArray,
    w_ar: FloatArray,
    q_ar: FloatArray,
    p_ar: FloatArray,
    r_ar: FloatArray,
    trim_conditions: FloatArray,
):
    """
    This function calculates the stability derivatives of the airplane using the finite difference method.
    For each of the six degrees of freedom, the function calculates the stability derivatives for the
    positive and negative perturbations of the state variable. The function then saves the results in
    a file named after the perturbation and the state variable. Variables:
        -u: Array of perturbations in the x-axis velocity.
        -v: Array of perturbations in the y-axis velocity.
        -w: Array of perturbations in the z-axis velocity.
        -q: Array of perturbations in the pitch rate.
        -p: Array of perturbations in the roll rate.
        -r: Array of perturbations in the yaw rate.

    Args:
        plane (Airplane): Airplane object.
        u_ar (FloatArray): Array of perturbations in the x-axis velocity. The perturbations are taken as central differences around the triim.
        v_ar (FloatArray): Array of perturbations in the y-axis velocity. The perturbations are taken as central differences around the triim.
        w_ar (FloatArray): Array of perturbations in the z-axis velocity. The perturbations are taken as central differences around the triim.
        q_ar (FloatArray): Array of perturbations in the pitch rate. The perturbations are taken as central differences around the triim.
        p_ar (FloatArray): Array of perturbations in the roll rate. The perturbations are taken as central differences around the triim.
        r_ar (FloatArray): Array of perturbations in the yaw rate. The perturbations are taken as central differences around the triim.
        trim_conditions (FloatArray): Array containing the trim conditions of the airplane. The array is of the form [alpha, V].
    """
    pl_dir = f"{plane.name}_genD"
    dif_path = f"{plane.name}_difs"
    case_num = 0

    u_ar = np.array(u_ar)
    v_ar = np.array(v_ar)
    w_ar = np.array(w_ar)
    q_ar = np.array(q_ar)
    p_ar = np.array(p_ar)
    r_ar = np.array(r_ar)

    # Check for empty arrays
    for arr in [u_ar, v_ar, w_ar, q_ar, p_ar, r_ar]:
        if arr.size == 0:
            raise ValueError("Empty array")

    os.makedirs(dif_path, exist_ok=True)

    for w in w_ar:
        li = []
        li.append(f"load {pl_dir}/{plane.name}.avl")
        li.append(f"mass {pl_dir}/{plane.name}.mass")
        li.append("MSET")
        li.append("0")
        li.append("oper")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        f_ang = trim_conditions[0] + rad2deg(np.arctan(w / trim_conditions[1]))
        b_ang = trim_conditions[0] - rad2deg(np.arctan(w / trim_conditions[1]))
        li.append(f"{f_ang}")
        li.append("x")
        li.append("FT")
        f_w = f"{plane.name}_dif_f_w_{w}"
        li.append(f_w)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1

        li.append("a")
        li.append("a")
        li.append(f"{b_ang}")
        li.append("x")
        li.append("FT")
        b_w = f"{plane.name}_dif_b_w_{w}"
        li.append(b_w)

        ar_1 = np.array(li)

    for u in u_ar:
        li = []
        w_0 = np.tan(deg2rad(trim_conditions[0])) * trim_conditions[1]
        f_u_ang = rad2deg(np.arctan(w_0 / (trim_conditions[1] + u)))
        b_u_ang = rad2deg(np.arctan(w_0 / (trim_conditions[1] - u)))
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{f_u_ang}")
        li.append("x")
        li.append("FT")
        f_u = f"{plane.name}_dif_f_u_{u}"
        li.append(f_u)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{b_u_ang}")
        li.append("x")
        li.append("FT")
        b_u = f"{plane.name}_dif_b_u_{u}"
        li.append(b_u)

        ar_2 = np.array(li)

        print(u)

    for q in q_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{(q*plane.mean_aerodynamic_chord/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_q = f"{plane.name}_dif_f_q_{q}"
        li.append(f_q)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{-(q*plane.mean_aerodynamic_chord/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_q = f"{plane.name}_dif_b_q_{q}"
        li.append(b_q)

        ar_3 = np.array(li)

    for v in v_ar:
        li = []

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("p")
        li.append("p")
        li.append(f"{0}")
        li.append("b")
        li.append("b")
        f_ang = rad2deg(np.arctan(v / trim_conditions[1]))
        b_ang = rad2deg(np.arctan(-v / trim_conditions[1]))
        li.append(f"{f_ang}")
        li.append("x")
        li.append("FT")
        f_v = f"{plane.name}_dif_f_v_{v}"
        li.append(f_v)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1

        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"{b_ang}")
        li.append("x")
        li.append("FT")
        b_v = f"{plane.name}_dif_b_v_{v}"
        li.append(b_v)
        ar_4 = np.array(li)

    for p in p_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"{(p*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_p = f"{plane.name}_dif_f_p_{p}"
        li.append(f_p)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"{-(p*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_p = f"{plane.name}_dif_b_p_{p}"
        li.append(b_p)
        ar_5 = np.array(li)

    for r in r_ar:
        li = []
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("r")
        li.append("r")
        li.append(f"0")
        li.append("y")
        li.append("y")
        li.append(f"{(r*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        f_r = f"{plane.name}_dif_f_r_{r}"
        li.append(f_r)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{trim_conditions[0]}")
        li.append("b")
        li.append("b")
        li.append(f"0")
        li.append("y")
        li.append("y")
        li.append(f"{-(r*plane.span/(2*trim_conditions[1]))}")
        li.append("x")
        li.append("FT")
        b_r = f"{plane.name}_dif_b_r_{r}"
        li.append(b_r)
        li.append("    ")
        li.append("quit")
        ar_6 = np.array(li)

    ar_6 = np.concatenate((ar_1, ar_2, ar_3, ar_4, ar_5, ar_6))
    np.savetxt(f"{plane.name}_difs_script", ar_6, delimiter=" ", fmt="%s")

    with open(f"{plane.name}_difs_script", encoding="utf-8") as fin:
        res: int = subprocess.check_call(
            [AVL_exe],
            stdin=fin,
        )

    for item in [
        b_q,
        f_q,
        b_u,
        f_u,
        b_w,
        f_w,
        b_p,
        f_p,
        b_v,
        f_v,
        b_r,
        f_r,
        f"{plane.name}_difs_script",
    ]:
        if os.path.isfile(item):
            # copy the file to the dif_path
            shutil.copy(item, dif_path)

            os.remove(item)


# LONGITUDAL STATE MATRIX USING FINITE DIFFERENCE STABILITY DERIVATIVES -
# EXECUTION OF POSTPRROCESSING FUNCTION NECESSARY
def long_mat(plane, environment, aoa_trim, trim_Vel, u_dict, I_li, w_dict, q_dict, span=4.4):
    mass = plane.M
    rho = environment.air_density
    g = environment.GRAVITY
    Area = plane.S
    chord = plane.mean_aerodynamic_chord
    span = plane.span
    # good inc = 0.005*U

    U_e = trim_Vel * np.cos(deg2rad(aoa_trim))
    W_e = trim_Vel * np.sin(deg2rad(aoa_trim))

    V_u_f = np.linalg.norm([u_dict["vals"][1] + U_e, W_e])
    V_u_b = np.linalg.norm([u_dict["vals"][0] + U_e, W_e])

    V_w_f = np.linalg.norm([U_e, W_e + w_dict["vals"][1]])
    V_w_b = np.linalg.norm([U_e, W_e + w_dict["vals"][1]])

    print(V_u_f**2 - V_u_b**2)

    X_u = (
        (u_dict["CX"][1] * V_u_f**2 - u_dict["CX"][0] * V_u_b**2)
        * (0.5 * rho * Area)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )
    Z_u = (
        (u_dict["CZ"][1] * V_u_f**2 - u_dict["CZ"][0] * V_u_b**2)
        * (0.5 * rho * Area)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )
    M_u = (
        (u_dict["Cm"][1] * V_u_f**2 - u_dict["Cm"][0] * V_u_b**2)
        * (0.5 * rho * Area * chord)
        / (u_dict["vals"][1] - u_dict["vals"][0])
    )

    X_w = (
        (w_dict["CX"][1] * V_w_f**2 - w_dict["CX"][0] * V_w_b**2)
        * (0.5 * rho * Area)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )
    Z_w = (
        (w_dict["CZ"][1] * V_w_f**2 - w_dict["CZ"][0] * V_w_b**2)
        * (0.5 * rho * Area)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )
    M_w = (
        (w_dict["Cm"][1] * V_w_f**2 - w_dict["Cm"][0] * V_w_b**2)
        * (0.5 * rho * Area * chord)
        / (w_dict["vals"][1] - w_dict["vals"][0])
    )

    X_q = (
        (q_dict["CX"][1] - q_dict["CX"][0])
        * (0.5 * rho * trim_Vel**2 * Area)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )
    Z_q = (
        (q_dict["CZ"][1] - q_dict["CZ"][0])
        * (0.5 * rho * trim_Vel**2 * Area)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )
    M_q = (
        (q_dict["Cm"][1] - q_dict["Cm"][0])
        * (0.5 * rho * trim_Vel**2 * Area * chord)
        / (q_dict["vals"][1] - q_dict["vals"][0])
    )

    state_mat = np.zeros((4, 4))

    state_mat[0, 0] = X_u / mass
    state_mat[1, 0] = Z_u / mass
    state_mat[2, 0] = M_u / I_li[1]

    state_mat[0, 1] = X_w / mass
    state_mat[1, 1] = Z_w / mass
    state_mat[2, 1] = M_w / I_li[1]

    state_mat[0, 2] = (X_q - mass * W_e) / mass
    state_mat[1, 2] = (Z_q + mass * U_e) / mass
    state_mat[2, 2] = M_q / I_li[1]
    state_mat[3, 0] = 0
    state_mat[3, 1] = 0
    state_mat[3, 2] = 1

    state_mat[0, 3] = -g
    state_mat[1, 3] = 0
    # print(Z_q)
    # print(-Z_q)

    return state_mat


# LATERAL STATE MATRIX USING FINITE DIFFERENCE STABILITY DERIVATIVES
def lat_mat(plane, environment, aoa_trim, trim_Vel, v_dict, I_li, p_dict, r_dict, span=4.4):
    # good inc = 0.005*U

    mass = plane.M
    rho = environment.air_density
    g = environment.GRAVITY
    Area = plane.S
    chord = plane.mean_aerodynamic_chord
    span = plane.span

    U_e = trim_Vel * np.cos(deg2rad(aoa_trim))
    W_e = trim_Vel * np.sin(deg2rad(aoa_trim))

    Y_v = (
        (v_dict["CY"][1] - v_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    l_v = (
        (v_dict["Cl"][1] - v_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    n_v = (
        (v_dict["Cn"][1] - v_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (v_dict["vals"][1] - v_dict["vals"][0])
    )

    Y_p = (
        (p_dict["CY"][1] - p_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    l_p = (
        (p_dict["Cl"][1] - p_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    n_p = (
        (p_dict["Cn"][1] - p_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (p_dict["vals"][1] - p_dict["vals"][0])
    )

    Y_r = (
        (r_dict["CY"][1] - r_dict["CY"][0])
        * (0.5 * rho * Area * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    l_r = (
        (r_dict["Cl"][1] - r_dict["Cl"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    n_r = (
        (r_dict["Cn"][1] - r_dict["Cn"][0])
        * (0.5 * rho * Area * span * trim_Vel**2)
        / (r_dict["vals"][1] - r_dict["vals"][0])
    )

    state_mat = np.zeros((4, 4))

    state_mat[0, 0] = Y_v / mass
    state_mat[1, 0] = (I_li[2] * l_v + (I_li[3]) * n_v) / (I_li[0] * I_li[2] - I_li[3] ** 2)
    state_mat[2, 0] = (I_li[0] * n_v + (I_li[3]) * l_v) / (I_li[0] * I_li[2] - I_li[3] ** 2)

    state_mat[0, 1] = (Y_p + mass * W_e) / mass
    state_mat[1, 1] = (I_li[2] * l_p + (I_li[3]) * n_p) / (I_li[0] * I_li[2] - I_li[3] ** 2)
    state_mat[2, 1] = (I_li[0] * n_p + -(I_li[3]) * l_p) / (I_li[0] * I_li[2] - I_li[3] ** 2)

    state_mat[0, 2] = (Y_r - mass * U_e) / mass
    state_mat[1, 2] = (I_li[2] * l_r + (I_li[3]) * n_r) / (I_li[0] * I_li[2] - I_li[3] ** 2)
    state_mat[2, 2] = (I_li[0] * n_r + (I_li[3]) * l_r) / (I_li[0] * I_li[2] - I_li[3] ** 2)

    state_mat[3, 0] = 0
    state_mat[3, 1] = 1
    state_mat[3, 2] = 0
    state_mat[3, 3] = 0

    state_mat[0, 3] = g
    state_mat[1, 3] = 0

    return state_mat
