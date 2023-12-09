import numpy as np
import os
import shutil


# FUNCTIONS FOR POLAR RUN - AFTER THE USER DEFINES ARRAY OF ANGLES
# THE CORRECT ORDER OF EXECUTION IS CASE_DEF - CASE_SETUP - CASE_RUN


# DEFINING ALL AOA RUNS IN .RUN AVL FILE
def case_def(plane, angles):
    polar_dir = f"{plane.name}_polarD"
    if os.path.isdir(polar_dir):
        shutil.rmtree(polar_dir)
    os.system(f"mkdir {polar_dir}")

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
    np.savetxt(f"{polar_dir}/{plane.name}.run", ar, delimiter=" ", fmt="%s")


# COREECT SETTING OF THE RUNS
def case_setup(plane):
    polar_dir = f"{plane.name}_polarD"
    pl_dir = f"{plane.name}_genD"
    li = []

    li.append(f"load {pl_dir}/{plane.name}.avl")
    li.append(f"mass {pl_dir}/{plane.name}.mass")
    li.append(f"case {polar_dir}/{plane.name}.run")
    li.append(f"MSET")
    li.append("0")
    li.append(f"oper")
    li.append("s")
    li.append(f"{polar_dir}/{plane.name}.run")
    li.append("y")
    li.append("    ")
    li.append("quit")
    ar = np.array(li)
    np.savetxt(f"{polar_dir}/{plane.name}_1_script", ar, delimiter=" ", fmt="%s")
    os.system(f"./avl.exe < {polar_dir}/{plane.name}_1_script")


# EXECUTION
def case_run(plane, angles):
    polar_dir = f"{plane.name}_polarD"
    pl_dir = f"{plane.name}_genD"

    li_2 = []
    li_2.append(f"load {pl_dir}/{plane.name}.avl")
    li_2.append(f"mass {pl_dir}/{plane.name}.mass")
    li_2.append(f"case {polar_dir}/{plane.name}.run")
    li_2.append(f"MSET 0")
    li_2.append(f"oper")
    for i, angle in enumerate(angles):
        li_2.append(f"{i+1}")
        li_2.append(f"x")
        li_2.append(f"FT")
        if angle >= 0:
            li_2.append(f"{polar_dir}/{plane.name}_res_{angle}.txt")
        else:
            li_2.append(f"{polar_dir}/{plane.name}_res_M{np.abs(angle)}.txt")
        if os.path.isfile(f"{li_2[-1]}"):
            li_2.append("y")

        # li_2.append("y")
        # li.append(f"O")
    li_2.append("    ")
    li_2.append("quit")
    ar_2 = np.array(li_2)
    np.savetxt(f"{polar_dir}/{plane.name}_script", ar_2, delimiter=" ", fmt="%s")
    os.system(f"./avl.exe < {polar_dir}/{plane.name}_script")
