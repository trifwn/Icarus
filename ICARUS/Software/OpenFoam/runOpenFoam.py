import os
import shutil
from subprocess import call

import numpy as np
import pandas as pd
from pands import DataFrame

from ICARUS.Software import logOFscript
from ICARUS.Software import runOFscript
from ICARUS.Software import setupOFscript


def make_mesh(airfoilFile: str, airfoilName: str, OFBASE: str, HOMEDIR: str) -> None:
    os.chdir(OFBASE)
    call(["/bin/bash", "-c", f"{setupOFscript} -n {airfoilName} -p {airfoilFile}"])
    os.chdir(HOMEDIR)


def setup_open_foam(
    OFBASE: str,
    CASEDIR: str,
    HOMEDIR: str,
    airfoil_file: str,
    airfoil_name: str,
    reynolds: float,
    Mach: float,
    all_angles_deg: list[float],
    silent: bool = False,
    max_iterations: int = 5000,
) -> None:
    """Function to setup OpenFoam cases for a given airfoil and Reynolds number

    Args:
        OFBASE (str): Base directory for OpenFoam Mock Case
        CASEDIR (str): Directory of Current Case
        HOMEDIR (str): Home Directory
        airfoilFile (str): Filename containg the arifoil geometry
        airfoilname (str): Name of the airfoil
        Reynolds (float): Reynolds Number
        Mach (float): Mach Number
        all_angles_deg (list[float]): All angles to run simulations for
        silent (bool, optional): Whether to be verbose. Defaults to False.
        max_iterations (int, optional): Max iterations for the simulation. Defaults to 5000.
    """
    os.chdir(CASEDIR)
    make_mesh(airfoil_file, airfoil_name, OFBASE, HOMEDIR)
    for angle_deg in all_angles_deg:
        if angle_deg >= 0:
            folder: str = str(angle_deg)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle_deg)[::-1].strip("-").zfill(6)[::-1]
        ANGLEDIR: str = os.path.join(CASEDIR, folder)
        os.makedirs(ANGLEDIR, exist_ok=True)
        os.chdir(ANGLEDIR)
        angle: float = angle_deg * np.pi / 180

        # MAKE 0/ FOLDER
        src: str = os.path.join(OFBASE, "0")
        dst: str = os.path.join(ANGLEDIR, "0")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        filename: str = os.path.join(ANGLEDIR, "0", "U")
        with open(filename, encoding="UTF-8", newline="\n") as file:
            data: list[str] = file.readlines()
        data[26] = f"internalField uniform ( {np.cos(angle)} {np.sin(angle)} 0. );\n"
        with open(filename, "w", encoding="UTF-8") as file:
            file.writelines(data)

        # MAKE constant/ FOLDER
        src = os.path.join(OFBASE, "constant")
        dst = os.path.join(ANGLEDIR, "constant")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        filename = os.path.join(ANGLEDIR, "constant", "transportProperties")
        with open(filename, encoding="UTF-8", newline="\n") as file:
            data = file.readlines()
        data[
            20
        ] = f"nu              [0 2 -1 0 0 0 0] {np.format_float_scientific(1/reynolds,sign=False,precision=3)};\n"
        with open(filename, "w", encoding="UTF-8") as file:
            file.writelines(data)

        # MAKE system/ FOLDER
        src = os.path.join(OFBASE, "system")
        dst = os.path.join(ANGLEDIR, "system")
        shutil.copytree(src, dst, dirs_exist_ok=True)

        filename = os.path.join(ANGLEDIR, "system", "controlDict")
        with open(filename, encoding="UTF-8", newline="\n") as file:
            data = file.readlines()
        data[36] = f"endTime {max_iterations}.;\n"
        data[94] = "\t\tCofR  (0.25 0. 0.);\n"
        data[95] = f"\t\tliftDir ({-np.sin(angle)} {np.cos(angle)} {0.});\n"
        data[96] = f"\t\tdragDir ({np.cos(angle)} {np.sin(angle)} {0.});\n"
        data[97] = "\t\tpitchAxis (0. 0. 1.);\n"
        data[98] = "\t\tmagUInf 1.;\n"
        data[110] = f"\t\tUInf ({np.cos(angle)} {np.sin(angle)} {0.});\n"
        with open(filename, "w", encoding="UTF-8") as file:
            file.writelines(data)
        if silent is False:
            print(f"{ANGLEDIR} Ready to Run")
    os.chdir(HOMEDIR)


def run_angle(CASEDIR: str, angle: float) -> None:
    """Function to run OpenFoam for a given angle given it is already setup

    Args:
        CASEDIR (str): CASE DIRECTORY
        angle (float): Angle to run
    """
    if angle >= 0:
        folder: str = str(angle)[::-1].zfill(7)[::-1]
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]

    ANGLEDIR: str = os.path.join(CASEDIR, folder)
    os.chdir(ANGLEDIR)
    print(f"{angle} deg: Simulation Starting")
    call(["/bin/bash", "-c", f"{runOFscript}"])
    os.chdir(CASEDIR)


def run_multiple_angles(CASEDIR: str, HOMEDIR: str, anglesAll: list[float]) -> None:
    """Function to run OpenFoam for multiple angles given it is already setup

    Args:
        CASEDIR (str): _description_
        HOMEDIR (str): _description_
        anglesAll (list[float]): _description_
    """
    for angle in anglesAll:
        run_angle(CASEDIR, angle)
    os.chdir(HOMEDIR)


def make_polars(CASEDIR: str, HOMEDIR: str, all_angles: list[float]) -> DataFrame:
    """Function to make polars from OpenFoam results

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        all_angles (list[float]): Angles to make polars for

    Returns:
        DataFrame: Dataframe Containing CL, CD, CM for all angles
    """
    os.chdir(CASEDIR)
    cd: list[float] = []
    cl: list[float] = []
    cm: list[float] = []
    folders: list[str] = next(os.walk("."))[1]
    angles_succeded: list[float] = []
    for angle in all_angles:
        if angle >= 0:
            folder: str = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        if folder in folders:
            data = get_coefficients(angle)
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
                ) = (float(i) for i in data.split("\t"))
                angles_succeded.append(angle)
                cd.append(Cd)
                cl.append(Cl)
                cm.append(CmPitch)
    df: DataFrame = pd.DataFrame(
        np.vstack([angles_succeded, cl, cd, cm]).T,
        columns=["AoA", "CL", "CD", "CM"],
    ).sort_values("AoA")

    df.to_csv("clcd.of", index=False)
    os.chdir(HOMEDIR)
    return df


def get_coefficients(angle: float) -> str | None:
    """Function to get coefficients from OpenFoam results for a given angle.

    Args:
        angle (float): Angle for which coefficients are required

    Returns:
        str | None: String Containing Coefficients or None if not found
    """
    if angle >= 0:
        folder: str = str(angle)[::-1].zfill(7)[::-1]
    else:
        folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
    parentDir: str = os.getcwd()
    folders: list[str] = next(os.walk("."))[1]
    if folder[:-1] in folders:
        os.chdir(folder)
        folders = next(os.walk("."))[1]
    if "postProcessing" in folders:
        os.chdir(os.path.join("postProcessing", "force_coefs"))
        times: list[str] = next(os.walk("."))[1]
        times_num = [int(times[j]) for j in range(len(times)) if times[j].isdigit()]
        latestTime = max(times_num)
        os.chdir(str(latestTime))
        filen = "coefficient.dat"
        with open(filen, encoding="UTF-8", newline="\n") as file:
            data: list[str] = file.readlines()
        os.chdir(parentDir)
    else:
        os.chdir(parentDir)
        return None
    return data[-1]


def clean_open_foam(HOMEDIR: str, CASEDIR: str) -> None:
    """Function to clean OpenFoam results

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    os.chdir(CASEDIR)
    for item in next(os.walk("."))[1]:
        if item.startswith("m") or item.startswith(
            ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"),
        ):
            os.chdir(item)
            times: list[str] = next(os.walk("."))[1]
            times_num: list[int] = [
                int(times[j]) for j in range(len(times)) if times[j].isdigit()
            ]
            times_num = sorted(times_num)
            for delFol in times[1:-1]:
                os.rmdir(delFol)
            os.chdir(CASEDIR)
    os.chdir(HOMEDIR)


def get_convergence_data(HOMEDIR: str, CASEDIR: str) -> None:
    """Function to create convergence data for OpenFoam results using the FoamLog script

    Args:
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
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
#             os.chdir(os.path.join('postProcessing',force_coefs'))
#             times = next(os.walk("."))[1]
#             times = [int(times[j]) for j in range(len(times))
#                      if times[j].isdigit()]
#             print(max(times))

#             os.chdir(parentdir)
#             break
