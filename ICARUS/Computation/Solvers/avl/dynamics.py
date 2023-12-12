import os
import shutil
import subprocess
from io import StringIO

import numpy as np
from numpy import deg2rad
from numpy import floating
from numpy import rad2deg

from ICARUS.Computation.Solvers.AVL.input import make_input_files
from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Environment.definition import Environment
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def csplit(input_file: str, pattern: str) -> list[str]:
    with open(input_file) as file:
        content = file.read()

    import re

    sections = re.split(pattern, content)
    sections = [section.strip() for section in sections if section.strip()]

    return sections


# EIGENVALUE ANALYSIS BASED ON THE IMPLICIT DIFFERENTIATION APPROACH OF M.DRELA AND AVL
def implicit_eigs(
    PLANEDIR: str,
    plane: Airplane,
    environment: Environment,
    UINF: float,
    solver2D: str = "Xfoil",
):
    DYNAMICS_DIR = os.path.join(PLANEDIR, "Dynamics")
    HOMEDIR = os.getcwd()
    make_input_files(
        PLANEDIR=DYNAMICS_DIR,
        plane=plane,
        environment=environment,
        UINF=UINF,
        solver2D=solver2D,
    )
    log = os.path.join(DYNAMICS_DIR, "eig_log.txt")
    os.makedirs(DYNAMICS_DIR, exist_ok=True)

    f_li = []
    f_li.append(f"load {plane.name}.avl\n")
    f_li.append(f"mass {plane.name}.mass\n")
    f_li.append("MSET\n")
    f_li.append("0\n")
    f_li.append("oper\n")
    f_li.append("1\n")
    f_li.append("a\n")
    f_li.append("pm\n")
    f_li.append("0\n")
    f_li.append("x\n")
    f_li.append("C1\n")
    f_li.append("b\n")
    f_li.append("0\n")
    f_li.append("    \n")
    f_li.append("x\n")
    f_li.append("    \n")
    f_li.append("mode\n")
    f_li.append("1\n")
    f_li.append("N\n")
    f_li.append("    \n")
    f_li.append("quit\n")

    input_f = os.path.join(DYNAMICS_DIR, f"eig_mode_scr")
    os.chdir(DYNAMICS_DIR)
    with open(input_f, "w", encoding="utf-8") as f:
        f.writelines(f_li)

    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    os.chdir(HOMEDIR)
    sections = csplit(log, "1:")
    sec_2_use = sections[-1].splitlines()
    sec_2_use[0] = "  mode 1:" + sec_2_use[0]

    def get_matrix(
        index: int,
        lines: list[str],
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Extracts the EigenVector and EgienValue from AVL Output

        Args:
            index (int): Index in Reading File
            lines (list[str]): AVL output

        Returns:
            tuple[FloatArray, FloatArray, FloatArray]: Longitudal EigenVector, Lateral EigenVector, Mode
        """
        index = int(index)
        mode = np.array([float(lines[index][10:22]), float(lines[index][24:36])])
        long_vecs = np.zeros((4, 2), dtype=floating)
        lat_vecs = np.zeros((4, 2), dtype=floating)
        for i in range(0, 4):
            long_vecs[i, :] = [
                float(lines[index + i + 1][9:15]),
                float(lines[index + i + 1][20:26]),
            ]

            lat_vecs[i, :] = [
                float(lines[index + i + 1][41:47]),
                float(lines[index + i + 1][52:58]),
            ]

        return long_vecs, lat_vecs, mode

    longitudal_matrix = []
    lateral_matrix = []
    ind_li = np.arange(0, 8, 1) * 6
    for i in ind_li:
        long_vec, lat_vec, mode = get_matrix(i, sec_2_use)
        if np.mean(np.abs(long_vec)) == 0:
            lateral_matrix.append(mode)
        elif np.mean(np.abs(lat_vec)) == 0:
            longitudal_matrix.append(mode)

    return longitudal_matrix, lateral_matrix


def trim_calculation(PLANE_DIR: str, plane: Airplane) -> tuple[float, float]:
    """Calculates the trim conditions of the airplane.

    Args:
        PLANE_DIR (str): Path to the directory containing the airplane files.
        plane (Airplane): Airplane object.
    """
    f_io: StringIO = StringIO()
    f_io.write(f"load {plane.name}.avl\n")
    f_io.write(f"mass {plane.name}.mass\n")
    f_io.write(f"MSET\n")
    f_io.write("0\n")
    f_io.write(f"oper\n")
    f_io.write("1\n")
    f_io.write(f"a\n")
    f_io.write("pm\n")
    f_io.write("0\n")
    f_io.write("x\n")
    f_io.write("c1\n")
    f_io.write("b\n")
    f_io.write("0\n")
    f_io.write("    \n")
    f_io.write("x\n")
    f_io.write("FT\n")
    f_io.write(f"trim_res.txt\n")
    if os.path.isfile(os.path.join(PLANE_DIR, "trim_res.txt")):
        f_io.write("O\n")
    f_io.write("    \n")
    f_io.write("quit\n")

    input_f = os.path.join(PLANE_DIR, "trim_scr")
    log = os.path.join(PLANE_DIR, "trim_log.txt")

    contents: str = f_io.getvalue().expandtabs(4)
    with open(input_f, "w", encoding="utf-8") as file:
        file.write(contents)

    HOMEDIR = os.getcwd()
    os.chdir(PLANE_DIR)
    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    os.chdir(HOMEDIR)

    delimiter = "Setup"
    sections = csplit(log, delimiter)
    sec_2_use = sections[2].splitlines()
    # Use the extracted values as needed
    trim_velocity = float(sec_2_use[5][22:28])
    trim_aoa = float(sec_2_use[68][10:19])

    print("V:", trim_velocity)
    print("aoa:", trim_aoa)
    return trim_aoa, trim_velocity


# FUNCTION FOR THE CALCULATION OF STABILITY DERIVATIVES VIA FINITED DIFEERENCE METHOD
# This function calculates the stability derivatives of the airplane using the finite difference method.
# For each of the six degrees of freedom, the function calculates the stability derivatives for the
# positive and negative perturbations of the state variable. The function then saves the results in
# a file named after the perturbation and the state variable. Variables:
#     -u: Array of perturbations in the x-axis velocity.
#     -v: Array of perturbations in the y-axis velocity.
#     -w: Array of perturbations in the z-axis velocity.
#     -q: Array of perturbations in the pitch rate.
#     -p: Array of perturbations in the roll rate.
#     -r: Array of perturbations in the yaw rate.


def finite_difs(
    PLANEDIR: str,
    plane: Airplane,
    u_ar,
    v_ar,
    w_ar,
    q_ar,
    p_ar,
    r_ar,
    trim_conditions,
    state: State,
):
    DYNDIR = os.path.join(PLANEDIR, "Dynamics")
    HOMEDIR = os.getcwd()
    case_num = 0

    for i, dst in enumerate(state.disturbances):
        print(dst.name, dst.amplitude, dst.axis)

    os.makedirs(DYNDIR, exist_ok=True)

    for w in w_ar:
        li = []
        li.append(f"load {plane.name}.avl")
        li.append(f"mass {plane.name}.mass")
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
        f_w = f"dif_f_w_{w}"
        li.append(f_w)

        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1

        li.append("a")
        li.append("a")
        li.append(f"{b_ang}")
        li.append("x")
        li.append("FT")
        b_w = f"dif_b_w_{w}"
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
        f_u = f"dif_f_u_{u}"
        li.append(f_u)
        li.append("+")
        li.append(f"{int(case_num+1)}")
        case_num = case_num + 1
        li.append("a")
        li.append("a")
        li.append(f"{b_u_ang}")
        li.append("x")
        li.append("FT")
        b_u = f"dif_b_u_{u}"
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
        f_q = f"dif_f_q_{q}"
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
        b_q = f"dif_b_q_{q}"
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
        f_v = f"dif_f_v_{v}"
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
        b_v = f"dif_b_v_{v}"
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
        f_p = f"dif_f_p_{p}"
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
        b_p = f"dif_b_p_{p}"
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
        f_r = f"dif_f_r_{r}"
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
        b_r = f"dif_b_r_{r}"
        li.append(b_r)
        li.append("    ")
        li.append("quit")
        ar_6 = np.array(li)

    all_arrays = np.concatenate((ar_1, ar_2, ar_3, ar_4, ar_5, ar_6))
    input_f = os.path.join(DYNDIR, "diffs_script")
    log = os.path.join(DYNDIR, "finite_diffs_log")
    with open(input_f, "w", encoding="utf-8"):
        np.savetxt(input_f, all_arrays, delimiter=" ", fmt="%s")

    os.chdir(DYNDIR)
    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call(
                [AVL_exe],
                stdin=fin,  # stdout=fout, stderr=fout
            )
    os.chdir(HOMEDIR)
