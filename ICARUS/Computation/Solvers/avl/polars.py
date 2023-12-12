import os
import subprocess

import numpy as np

from ICARUS.Core.types import FloatArray
from ICARUS.Database import AVL_exe
from ICARUS.Database.utils import angle_to_case
from ICARUS.Vehicle.plane import Airplane


# FUNCTIONS FOR POLAR RUN - AFTER THE USER DEFINES ARRAY OF ANGLES
# THE CORRECT ORDER OF EXECUTION IS CASE_DEF - CASE_SETUP - CASE_RUN


# DEFINING ALL AOA RUNS IN .RUN AVL FILE
def case_def(PLANEDIR: str, plane: Airplane, angles: FloatArray) -> None:
    li = []
    for i, angle in enumerate(angles):
        li.append("---------------------------------------------")
        li.append(f"Run case  {i+1}:  -{angle}_deg- ")
        li.append(" ")
        li.append(f" alpha        ->  alpha       =  {angle}")
        li.append("beta         ->  beta        =   0.00000")
        li.append("pb/2V        ->  pb/2V       =   0.00000")
        li.append("qc/2V        ->  qc/2V       =   0.00000")
        li.append("rb/2V        ->  rb/2V       =   0.00000")
    ar = np.array(li)
    np.savetxt(os.path.join(PLANEDIR, f"{plane.name}.run"), ar, delimiter=" ", fmt="%s")


def case_setup(PLANEDIR: str, plane: Airplane) -> None:
    HOMEDIR = os.getcwd()
    os.chdir(PLANEDIR)

    li = []

    li.append(f"load {plane.name}.avl")
    li.append(f"mass {plane.name}.mass")
    li.append(f"case {plane.name}.run")
    li.append(f"MSET")
    li.append("0")
    li.append(f"oper")
    li.append("s")
    li.append(f"{plane.name}.run")
    li.append("y")
    li.append("    ")
    li.append("quit")
    ar = np.array(li)

    np.savetxt(f"mset_script", ar, delimiter=" ", fmt="%s")

    with open(f"mset_script") as fin:
        with open(f"mset_script.out", "w") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                stderr=fout,
            )
    os.chdir(HOMEDIR)


# EXECUTION
def case_run(PLANEDIR: str, plane: Airplane, angles: FloatArray) -> None:
    HOMEDIR = os.getcwd()
    os.chdir(PLANEDIR)
    li_2 = []
    li_2.append(f"load {plane.name}.avl")
    li_2.append(f"mass {plane.name}.mass")
    li_2.append(f"case {plane.name}.run")
    li_2.append(f"MSET 0")
    li_2.append(f"oper")
    for i, angle in enumerate(angles):
        li_2.append(f"{i+1}")
        li_2.append(f"x")
        li_2.append(f"FT")
        li_2.append(f"{angle_to_case(angle)}.txt")

        if os.path.isfile(f"{li_2[-1]}"):
            li_2.append("o")

        # li_2.append("y")
        # li.append(f"O")
    li_2.append("    ")
    li_2.append("quit")
    ar_2 = np.array(li_2)
    np.savetxt(f"polar_script", ar_2, delimiter=" ", fmt="%s")

    # Do the same with subprocess
    with open(f"polar_script") as fin:
        with open(f"polar_script.out", "w") as fout:
            res = subprocess.check_call(
                [AVL_exe],
                stdin=fin,
                stdout=fout,
                # stderr=fout,
            )
    print(f"AVL return code: {res}")
    os.chdir(HOMEDIR)
