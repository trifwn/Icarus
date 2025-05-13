import os
import subprocess
from io import StringIO
from typing import Literal

import numpy as np
from numpy import deg2rad
from numpy import rad2deg

from ICARUS import AVL_exe
from ICARUS.database import Database
from ICARUS.database.utils import disturbance_to_case
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.airplane import Airplane

from .input import make_input_files


def implicit_eigs(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
) -> None:
    # EIGENVALUE ANALYSIS BASED ON THE IMPLICIT DIFFERENTIATION APPROACH OF M.DRELA AND AVL
    DB = Database.get_instance()
    DYNAMICS_DIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )
    HOMEDIR = os.getcwd()
    os.makedirs(DYNAMICS_DIR, exist_ok=True)
    make_input_files(
        directory=DYNAMICS_DIR,
        plane=plane,
        state=state,
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

    input_f = os.path.join(DYNAMICS_DIR, "eig_mode_scr")
    os.chdir(DYNAMICS_DIR)
    with open(input_f, "w", encoding="utf-8") as f:
        f.writelines(f_li)

    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            _: int = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    os.chdir(HOMEDIR)


def finite_difs(
    plane: Airplane,
    state: State,
    solver2D: Literal["Xfoil", "Foil2Wake", "OpenFoam"] | str = "Xfoil",
) -> None:
    DB = Database.get_instance()
    DYNAMICS_DIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )
    HOMEDIR = os.getcwd()
    os.makedirs(DYNAMICS_DIR, exist_ok=True)
    make_input_files(
        directory=DYNAMICS_DIR,
        plane=plane,
        state=state,
        solver2D=solver2D,
    )
    os.chdir(DYNAMICS_DIR)

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

    w_velocity = np.tan(deg2rad(state.trim["AoA"])) * state.trim["U"]
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
            continue
        else:
            print(f"Got unexpected var {dst.var}")
            continue

        f_io.write(f"{int(i + 1)}\n")
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

    input_f = os.path.join(DYNAMICS_DIR, "diffs_script")
    log = os.path.join(DYNAMICS_DIR, "finite_diffs_log")
    contents = f_io.getvalue().expandtabs(4)
    with open(input_f, "w", encoding="utf-8") as f:
        f.write(contents)

    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            _: int = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    os.chdir(HOMEDIR)
