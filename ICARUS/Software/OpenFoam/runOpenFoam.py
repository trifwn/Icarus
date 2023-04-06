from subprocess import call
import os
import shutil
import numpy as np
import pandas as pd
from ICARUS.Software import runOFscript, setupOFscript, logOFscript


def makeMesh(airfoilFile, airfoilName, OFBASE, HOMEDIR):
    os.chdir(OFBASE)
    call(
        ["/bin/bash", "-c", f"{setupOFscript} -n {airfoilName} -p {airfoilFile}"])
    os.chdir(HOMEDIR)


def setupOpenFoam(OFBASE, CASEDIR, HOMEDIR, airfoilFile,
                  airfoilname, Reynolds, Mach, anglesALL,
                  silent=False, maxITER=5000):
    os.chdir(CASEDIR)
    makeMesh(airfoilFile, airfoilname, OFBASE, HOMEDIR)
    for ang in anglesALL:
        if ang >= 0:
            folder = str(ang)[::-1].zfill(7)[::-1] + "/"
        else:
            folder = "m" + str(ang)[::-1].strip("-").zfill(6)[::-1] + "/"
        ANGLEDIR = f"{CASEDIR}/{folder}"
        os.system(f"mkdir -p {ANGLEDIR}")
        os.chdir(ANGLEDIR)
        ang = ang * np.pi / 180

        # MAKE 0/ FOLDER
        shutil.copytree(f"{OFBASE}/0/", ANGLEDIR +
                        "/0/", dirs_exist_ok=True)
        filen = f"{ANGLEDIR}/0/U"
        with open(filen, "r", newline="\n") as file:
            data = file.readlines()
        data[26] = f"internalField uniform ( {np.cos(ang)} {np.sin(ang)} 0. );\n"
        with open(filen, "w") as file:
            file.writelines(data)

        # MAKE constant/ FOLDER
        shutil.copytree(f"{OFBASE}/constant/", ANGLEDIR +
                        "/constant/", dirs_exist_ok=True)
        filen = f"{ANGLEDIR}/constant/transportProperties"
        with open(filen, "r", newline="\n") as file:
            data = file.readlines()
        data[20] = f"nu              [0 2 -1 0 0 0 0] \
            {np.format_float_scientific(1/Reynolds,sign=False,precision=3)};\n"
        with open(filen, "w") as file:
            file.writelines(data)

        # MAKE system/ FOLDER
        shutil.copytree(f"{OFBASE}/system/", ANGLEDIR +
                        "/system/", dirs_exist_ok=True)
        filen = f"{ANGLEDIR}/system/controlDict"
        with open(filen, "r", newline="\n") as file:
            data = file.readlines()
        data[36] = f"endTime {maxITER}.;\n"
        data[94] = f"\t\tCofR  (0.25 0. 0.);\n"
        data[95] = f"\t\tliftDir ({-np.sin(ang)} {np.cos(ang)} {0.});\n"
        data[96] = f"\t\tdragDir ({np.cos(ang)} {np.sin(ang)} {0.});\n"
        data[97] = f"\t\tpitchAxis (0. 0. 1.);\n"
        data[98] = "\t\tmagUInf 1.;\n"
        data[110] = f"\t\tUInf ({np.cos(ang)} {np.sin(ang)} {0.});\n"
        with open(filen, "w") as file:
            file.writelines(data)
        if silent is False:
            print(f"{ANGLEDIR} Ready to Run")
    os.chdir(HOMEDIR)


def runFoamAngle(CASEDIR, angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] + "/"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"

    ANGLEDIR = f"{CASEDIR}/{folder}"
    os.chdir(ANGLEDIR)
    print(f'{angle} deg: Simulation Starting')
    os.system(f"{runOFscript}")
    os.chdir(CASEDIR)


def runFoam(CASEDIR, HOMEDIR, anglesAll):
    for angle in anglesAll:
        runFoamAngle(CASEDIR, angle)
    os.chdir(HOMEDIR)


def makeCLCD(CASEDIR, HOMEDIR, anglesAll):
    os.chdir(CASEDIR)
    cd = []
    cl = []
    cm = []
    folders = next(os.walk("."))[1]
    angleSucc = []
    for angle in anglesAll:
        if angle >= 0:
            folder = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        if folder in folders:
            data = getCoeffs(angle)
            if data is not None:
                (
                    Time,
                    Cd,
                    Cdf,
                    Cdr,
                    Cl,
                    Clf,
                    Clr,
                    CmPitch,
                    CmRoll,
                    CmYaw,
                    Cs,
                    Csf,
                    Csr,
                ) = [float(i) for i in data.split("\t")]
                angleSucc.append(angle)
                cd.append(Cd)
                cl.append(Cl)
                cm.append(CmPitch)
    df = pd.DataFrame(np.vstack([angleSucc, cl, cd, cm]).T,
                      columns=["AoA", 'CL', "CD", "CM"])
    df = df.sort_values("AoA")
    df.to_csv('clcd.of', index=False)
    os.chdir(HOMEDIR)
    return df


def getCoeffs(angle):
    if angle >= 0:
        folder = str(angle)[::-1].zfill(7)[::-1] + "/"
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1] + "/"
    parentDir = os.getcwd()
    folders = next(os.walk("."))[1]
    if folder[:-1] in folders:
        os.chdir(folder)
        folders = next(os.walk("."))[1]
    if "postProcessing" in folders:
        os.chdir("postProcessing/force_coefs/")
        times = next(os.walk("."))[1]
        times = [int(times[j]) for j in range(len(times))
                 if times[j].isdigit()]
        latestTime = max(times)
        os.chdir(str(latestTime))
        filen = "coefficient.dat"
        with open(filen, "r", newline="\n") as file:
            data = file.readlines()
        os.chdir(parentDir)
    else:
        os.chdir(parentDir)
        return None
    return data[-1]


def cleanOpenFoam(HOMEDIR, CASEDIR):
    os.chdir(CASEDIR)
    for item in next(os.walk('.'))[1]:
        if item.startswith('m') or item.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            os.chdir(item)
            times = next(os.walk("."))[1]
            times = [int(times[j]) for j in range(len(times))
                     if times[j].isdigit()]
            times = sorted(times)
            for delFol in times[1:-1]:
                os.system(f"rm -r {delFol}")
            os.chdir(CASEDIR)
    os.chdir(HOMEDIR)


def getConvergence(HOMEDIR, CASEDIR):
    os.chdir(CASEDIR)
    call(["/bin/bash", "-c", f"{logOFscript}"])
    os.chdir("logs")

    os.chdir(HOMEDIR)

# def reorderFoamResults(anglesAll):
#     folders = next(os.walk("."))[1]
#     parentdir = os.getcwd()
#     for angle in anglesAll:
#         if angle >= 0:
#             folder = str(angle)[::-1].zfill(7)[::-1]
#         else:
#             folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
#         if folder in folders:
#             os.chdir(folder)
#             os.chdir('postProcessing/force_coefs')
#             times = next(os.walk("."))[1]
#             times = [int(times[j]) for j in range(len(times))
#                      if times[j].isdigit()]
#             print(max(times))

#             os.chdir(parentdir)
#             break
