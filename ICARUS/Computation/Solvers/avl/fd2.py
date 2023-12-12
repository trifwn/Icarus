import os
import subprocess
from io import StringIO

import numpy as np
from numpy import deg2rad
from numpy import rad2deg

from ICARUS.Database import AVL_exe
from ICARUS.Database.utils import disturbance_to_case
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def finite_difs(
    PLANEDIR: str,
    plane: Airplane,
    state: State,
):
    DYNDIR = os.path.join(PLANEDIR, "Dynamics")
    HOMEDIR = os.getcwd()
    os.makedirs(DYNDIR, exist_ok=True)
    os.chdir(DYNDIR)

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
            continue
        elif dst.var == "u":
            aoa = rad2deg(np.arctan(w_velocity / U + dst.amplitude))
        elif dst.var == "w":
            aoa += rad2deg(np.arctan(dst.amplitude / U))
        elif dst.var == "q":
            pitch_rate = dst.amplitude * plane.mean_aerodynamic_chord / (2 * U)
        elif dst.var == "theta":
            aoa += dst.amplitude
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
            f_io.write('o\n')
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
