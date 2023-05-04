import os
import shutil
import stat
import subprocess
from time import sleep

import numpy as np
import pandas as pd

from . import filesF2w as ff2w


def anglesSep(anglesALL):
    pangles = []
    nangles = []
    for ang in anglesALL:
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
            nangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles


def makeCLCD(CASEDIR, HOMEDIR, Reynolds, Mach):
    os.chdir(CASEDIR)
    folders = next(os.walk("."))[1]
    print("Making Polars")
    with open("output_bat", "w") as file:
        folder = folders[0]
        file.writelines("cd " + folder + "\n../write_out\n")
        for folder in folders[1:]:
            if "AERLOAD.OUT" in next(os.walk(folder))[2]:
                file.writelines("cd ../" + folder + "\n../write_out\n")

        # Write Cat command
        file.writelines("cd ..\n")
        file.writelines(
            'echo "Reynolds: '
            + str(Reynolds)
            + "\\nMach: "
            + str(Mach)
            + '" > clcd.out\n',
        )
        file.writelines("cat ")
        for folder in folders[::-1]:
            if "AERLOAD.OUT" in next(os.walk(folder))[2]:
                file.writelines(folder + "/clcd.out ")
        file.writelines(">> clcd.f2w")
    st = os.stat("output_bat")
    os.chmod("output_bat", st.st_mode | stat.S_IEXEC)
    subprocess.call([os.path.join(CASEDIR, "output_bat")])

    with open("clcd.f2w") as file:
        data = file.readlines()
    data = data[2:]
    nums = []
    for item in data:
        n = item.split()
        nums.append([float(i) for i in n])
    clcd = np.array(nums)
    os.chdir(HOMEDIR)
    return clcd


def makeCLCD2(CASEDIR, HOMEDIR):
    os.chdir(CASEDIR)
    folders = next(os.walk("."))[1]
    print("Making Polars")
    with open("output_bat", "w") as file:
        n = 0
        for folder in folders[1:]:
            if "AERLOAD.OUT" in next(os.walk(folder))[2]:
                if n == 0:
                    file.writelines("cd " + folder + "\n../write_out\n")
                    n += 1
                else:
                    file.writelines("cd ../" + folder + "\n../write_out\n")
    st = os.stat("output_bat")
    os.chmod("output_bat", st.st_mode | stat.S_IEXEC)
    subprocess.call([os.path.join(CASEDIR, "output_bat")])

    folders = next(os.walk("."))[1]
    a = []
    for folder in folders[1:]:
        if "clcd.out" in next(os.walk(folder))[2]:
            fileLOC = os.path.joint(folder, "clcd.out")
            a.append(np.loadtxt(fileLOC))
    df = pd.DataFrame(a, columns=["AoA", "CL", "CD", "Cm"])
    df = df.sort_values("AoA")
    df.to_csv("clcd.f2w", index=False)
    os.chdir(HOMEDIR)
    return df


def runF2W(CASEDIR, HOMEDIR, Reynolds, Mach, ftripL, ftripU, anglesALL, airfile):
    os.chdir(CASEDIR)
    nangles, pangles = anglesSep(anglesALL)
    for angles, name in [[pangles, "pos"], [nangles, "neg"]]:
        NoA = len(angles)

        # IO FILES
        ff2w.iofile(airfile)

        # DESIGN.INP
        ff2w.designFile(NoA, angles, name)

        # F2W.INP
        ff2w.f2winp(Reynolds, Mach, ftripL, ftripU, name)

        # RUN Files
        shutil.copy(f"design_{name}.inp", "design.inp")
        shutil.copy(f"f2w_{name}.inp", "f2w.inp")
        print(f"Running {angles}")
        f = open(f"{name}.out", "w")
        subprocess.call([os.path.join(CASEDIR, "foil_section")], stdout=f, stderr=f)
        os.rmdir("TMP.dir")
    os.remove("SOLOUTI*")
    sleep(1)
    # return makeCLCD(Reynolds,Mach)
    os.chdir(HOMEDIR)
    return 0


def removeResults(CASEDIR, HOMEDIR, angles):
    os.chdir(CASEDIR)
    os.remove("SOLOUTI*")
    os.remove("*.out")
    os.remove("PAKETO")
    parentDir = os.getcwd()
    folders = next(os.walk("."))[1]
    for angle in angles:
        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        if folder[:-1] in folders:
            os.chdir(folder)
            os.remove(
                "AERLOAD.OUT AIRFOIL.OUT BDLAYER.OUT COEFPRE.OUT SEPWAKE.OUT TREWAKE.OUT clcd.out SOLOUTI.INI",
            )
            os.chdir(parentDir)
    os.chdir(HOMEDIR)


def setupF2W(F2WBASE, HOMEDIR, CASEDIR):
    filesNeeded = [
        "design.inp",
        "design_neg.inp",
        "design_pos.inp",
        "f2w.inp",
        "f2w_neg.inp",
        "f2w_pos.inp",
        "io.files",
        "write_out",
    ]
    for item in filesNeeded:
        src = os.path.join(F2WBASE, item)
        dst = os.path.join(CASEDIR, item)
        shutil.copy(src, dst)
    if "foil" not in next(os.walk(CASEDIR))[2]:
        src = os.path.join(HOMEDIR, "ICARUS", "foil_section")
        dst = os.path.join(CASEDIR, "foil_section")
        os.symlink(src, dst)
