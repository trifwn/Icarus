import os
import shutil
import stat
import subprocess
from io import TextIOWrapper
from os import stat_result
from time import sleep
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame

from . import filesF2w as ff2w


def separate_angles(all_angles: list[float]) -> tuple[list[float], list[float]]:
    """Given A list of angles it separates them in positive and negative

    Args:
        all_angles (list[float]): Angles to separate

    Returns:
        tuple[list[float], list[float]]: Tuple of positive and negative angles
    """
    pangles: list[float] = []
    nangles: list[float] = []
    for ang in all_angles:
        if ang > 0:
            pangles.append(ang)
        elif ang == 0:
            pangles.append(ang)
            nangles.append(ang)
        else:
            nangles.append(ang)
    nangles = nangles[::-1]
    return nangles, pangles


def make_2d_polars(
    CASEDIR: str,
    HOMEDIR: str,
) -> ndarray[Any, dtype[floating[Any]]]:
    """
    Make the polars from the forces and return an np array with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory

    Returns:
        ndarray[Any, dtype[floating[Any]]]: Polars
    """
    os.chdir(CASEDIR)
    folders: list[str] = next(os.walk("."))[1]
    print("Making Polars")
    with open("output_bat", "w", encoding="utf-8") as file:
        folder: str = folders[0]
        file.writelines("cd " + folder + "\n../write_out\n")
        for folder in folders[1:]:
            if "AERLOAD.OUT" in next(os.walk(folder))[2]:
                file.writelines("cd ../" + folder + "\n../write_out\n")

        # Write Cat command
        file.writelines("cd ..\n")
        file.writelines("cat ")
        for folder in folders[::-1]:
            if "AERLOAD.OUT" in next(os.walk(folder))[2]:
                file.writelines(folder + "/clcd.out ")
        file.writelines(">> clcd.f2w")
    st: stat_result = os.stat("output_bat")
    os.chmod("output_bat", st.st_mode | stat.S_IEXEC)
    subprocess.call([os.path.join(CASEDIR, "output_bat")])

    with open("clcd.f2w", encoding="utf-8") as file:
        data: list[str] = file.readlines()
    data = data[2:]
    nums: list[list[float]] = []
    for item in data:
        n: list[str] = item.split()
        nums.append([float(i) for i in n])
    clcd: ndarray[Any, dtype[floating[Any]]] = np.array(nums)
    os.chdir(HOMEDIR)
    return clcd


def make_2d_polars_2(CASEDIR: str, HOMEDIR: str) -> DataFrame:
    """
    Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory

    Returns:
        DataFrame: Dataframe Containing CL, CD, CM for all angles
    """
    os.chdir(CASEDIR)
    folders: list[str] = next(os.walk("."))[1]
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
    st: stat_result = os.stat("output_bat")
    os.chmod("output_bat", st.st_mode | stat.S_IEXEC)
    subprocess.call([os.path.join(CASEDIR, "output_bat")])

    folders = next(os.walk("."))[1]
    a: list[Any] = []
    for folder in folders[1:]:
        if "clcd.out" in next(os.walk(folder))[2]:
            file_location: str = os.path.join(folder, "clcd.out")
            a.append(np.loadtxt(file_location))
    df: DataFrame = pd.DataFrame(a, columns=["AoA", "CL", "CD", "Cm"]).sort_values(
        "AoA",
    )
    df.to_csv("clcd.f2w", index=False)
    os.chdir(HOMEDIR)
    return df


def runF2W(
    CASEDIR: str,
    HOMEDIR: str,
    reynolds: float,
    mach: float,
    f_trip_low: dict[str, float],
    f_trip_upper: dict[str, float],
    all_angles: list[float],
    airfile: str,
) -> None:
    """
    Runs the f2w program for a given case making all necessary files

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        reynolds (float): Reynolds number
        mach (float): Mach number
        f_trip_low (dict[str, float]): Transition points for positive and negative angles for the lower surface
        f_trip_upper (dict[str, float]): Transition points for positive and negative angles for the upper surface
        all_angles (list[float]): All angles to run
        airfile (str): Name of the file containing the airfoil geometry
    """
    os.chdir(CASEDIR)
    nangles, pangles = separate_angles(all_angles)
    for angles, name in zip([pangles, nangles], ["pos", "neg"]):
        num_of_angles: int = len(angles)

        # IO FILES
        ff2w.io_file(airfile)

        # DESIGN.INP
        ff2w.desing_file(num_of_angles, angles, name)

        # F2W.INP
        ff2w.input_file(reynolds, mach, f_trip_low, f_trip_upper, name)

        # RUN Files
        shutil.copy(f"design_{name}.inp", "design.inp")
        shutil.copy(f"f2w_{name}.inp", "f2w.inp")
        print(f"Running {angles}")
        with open(f"{name}.out", "w") as f:
            subprocess.call([os.path.join(CASEDIR, "foil_section")], stdout=f, stderr=f)
        os.rmdir("TMP.dir")
    os.remove("SOLOUTI*")
    sleep(1)
    # return makeCLCD(Reynolds,Mach)
    os.chdir(HOMEDIR)


def remove_results(CASEDIR: str, HOMEDIR: str, angles: list[float]) -> None:
    """
    Removes Simulation results for a given case

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        angles (list[float]): Angles to remove
    """
    os.chdir(CASEDIR)
    os.remove("SOLOUTI*")
    os.remove("*.out")
    os.remove("PAKETO")
    parent_directory: str = os.getcwd()
    folders: list[str] = next(os.walk("."))[1]
    for angle in angles:
        if angle >= 0:
            folder: str = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
        if folder[:-1] in folders:
            os.chdir(folder)
            os.remove(
                "AERLOAD.OUT AIRFOIL.OUT BDLAYER.OUT COEFPRE.OUT SEPWAKE.OUT TREWAKE.OUT clcd.out SOLOUTI.INI",
            )
            os.chdir(parent_directory)
    os.chdir(HOMEDIR)


def setup_f2w(F2WBASE: str, HOMEDIR: str, CASEDIR: str) -> None:
    """
    Sets up the f2w case copying and editing all necessary files

    Args:
        F2WBASE (str): Base Case Directory for f2w
        HOMEDIR (str): Home Directory
        CASEDIR (str): Case Directory
    """
    filesNeeded: list[str] = [
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
        src: str = os.path.join(F2WBASE, item)
        dst: str = os.path.join(CASEDIR, item)
        shutil.copy(src, dst)
    if "foil" not in next(os.walk(CASEDIR))[2]:
        src = os.path.join(HOMEDIR, "ICARUS", "foil_section")
        dst = os.path.join(CASEDIR, "foil_section")
        os.symlink(src, dst)
