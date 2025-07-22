import os
import subprocess

from ICARUS import AVL_exe
from ICARUS.database import Database
from ICARUS.flight_dynamics import State
from ICARUS.solvers.AVL import AVLParameters
from ICARUS.vehicle import Airplane

from .input import make_input_files


def implicit_eigs(
    plane: Airplane,
    state: State,
    solver_parameters: AVLParameters = AVLParameters(),
) -> None:
    # EIGENVALUE ANALYSIS BASED ON THE IMPLICIT DIFFERENTIATION APPROACH OF M.DRELA AND AVL
    DB = Database.get_instance()
    DYNAMICS_DIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver="AVL",
        case="Dynamics",
    )
    os.makedirs(DYNAMICS_DIR, exist_ok=True)
    make_input_files(
        directory=DYNAMICS_DIR,
        plane=plane,
        state=state,
        solver_parameters=solver_parameters,
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
    with open(input_f, "w", encoding="utf-8") as f:
        f.writelines(f_li)

    with open(input_f, encoding="utf-8") as fin:
        with open(log, "w", encoding="utf-8") as fout:
            _: int = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
                cwd=DYNAMICS_DIR,
            )
