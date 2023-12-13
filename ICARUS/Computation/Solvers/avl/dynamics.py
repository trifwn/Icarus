import os
import shutil
import subprocess
from io import StringIO

import numpy as np
from numpy import deg2rad
from numpy import rad2deg

from ICARUS.Computation.Solvers.AVL.input import make_input_files
from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Database import DB3D
from ICARUS.Database.utils import disturbance_to_case
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
) -> tuple[list[complex], list[complex]]:
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
    sec_2_use[0] = "  mode 1:  " + sec_2_use[0]

    def get_matrix(
        index: int,
        lines: list[str],
    ) -> tuple[FloatArray, FloatArray, complex]:
        """Extracts the EigenVector and EgienValue from AVL Output

        Args:
            index (int): Index in Reading File
            lines (list[str]): AVL output

        Returns:
            tuple[FloatArray, FloatArray, FloatArray]: Longitudal EigenVector, Lateral EigenVector, Mode
        """
        mode = complex(float(lines[index][10:22]), float(lines[index][24:36]))
        long_vecs = np.zeros((4), dtype=complex)
        lat_vecs = np.zeros((4), dtype=complex)
        for i in range(0, 4):
            long_vecs[i] = complex(float(lines[index + i + 1][8:18]), float(lines[index + i + 1][19:28]))

            lat_vecs[i] = complex(
                float(lines[index + i + 1][40:50]),
                float(lines[index + i + 1][51:60]),
            )

        return long_vecs, lat_vecs, mode

    longitudal_matrix = []
    lateral_matrix = []
    indexes = np.arange(0, 8, 1) * 6
    for i in indexes:
        long_vec, lat_vec, mode = get_matrix(i, sec_2_use)
        if np.mean(np.abs(long_vec)) < np.mean(np.abs(lat_vec)):
            lateral_matrix.append(mode)
        else:
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


def finite_difs(
    plane: Airplane,
    state: State,
):
    DYNDIR = os.path.join(DB3D, plane.directory, "AVL", "Dynamics")
    HOMEDIR = os.getcwd()
    os.makedirs(DYNDIR, exist_ok=True)
    os.chdir(DYNDIR)

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
    f_io = StringIO()
    f_io.write(f"load {plane.name}.avl\n")
    f_io.write(f"mass {plane.name}.mass\n")
    f_io.write("MSET\n")
    f_io.write("0\n")
    f_io.write("oper\n")

    w_velocity = np.tan(rad2deg(state.trim["AoA"])) * state.trim["U"]
    U = state.trim["U"]
    for i, dst in enumerate(state.disturbances):
        aoa = state.trim["AoA"]
        beta = 0.0
        pitch_rate = 0.0
        roll_rate = 0.0
        yaw_rate = 0.0

        if dst.amplitude is None:
            pass
        elif dst.var == "u":
            aoa = rad2deg(np.arctan(w_velocity / (U + dst.amplitude)))
        elif dst.var == "w":
            aoa = aoa + rad2deg(np.arctan(dst.amplitude / U))
        elif dst.var == "q":
            pitch_rate = dst.amplitude * plane.mean_aerodynamic_chord / (2 * U)
        elif dst.var == "theta":
            continue
            # aoa = aoa + dst.amplitude
        elif dst.var == "v":
            beta = rad2deg(np.arctan(dst.amplitude / U))
        elif dst.var == "p":
            roll_rate = dst.amplitude * plane.span / (2 * U)
        elif dst.var == "r":
            yaw_rate = dst.amplitude * plane.span / (2 * U)
        elif dst.var == "phi":
            pass
            continue
        else:
            print(f"Got unexpected var {dst.var}")
            continue

        f_io.write(f"{int(i+1)}\n")
        f_io.write("a\n")
        f_io.write("a\n")
        f_io.write(f"{aoa}\n")

        f_io.write("b\n")
        f_io.write("b\n")
        f_io.write(f"{beta}\n")

        f_io.write("r\n")
        f_io.write("r\n")
        f_io.write(f"{roll_rate}\n")

        f_io.write("p\n")
        f_io.write("p\n")
        f_io.write(f"{pitch_rate}\n")

        f_io.write("y\n")
        f_io.write("y\n")
        f_io.write(f"{yaw_rate}\n")

        # EXECUTE CASE
        f_io.write("x\n")
        f_io.write("FT\n")
        case = disturbance_to_case(dst)
        f_io.write(f"{case}\n")
        if os.path.isfile(case):
            f_io.write("o\n")
        f_io.write("+\n")
    f_io.write("    \n")
    f_io.write("quit\n")

    input_f = os.path.join(DYNDIR, "diffs_script")
    log = os.path.join(DYNDIR, "finite_diffs_log")
    contents = f_io.getvalue().expandtabs(4)
    with open(input_f, "w", encoding="utf-8") as f:
        f.write(contents)

    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            res: int = subprocess.check_call([AVL_exe], stdin=fin, stdout=fout, stderr=fout)
    os.chdir(HOMEDIR)
