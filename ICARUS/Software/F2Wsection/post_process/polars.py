import numpy as np
from numpy import dtype, floating, ndarray
import pandas as pd
from pandas import DataFrame


import os
import stat
import subprocess
from os import stat_result
from typing import Any

def make_polars(CASEDIR: str, HOMEDIR: str) -> DataFrame:
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
    # print(f"Making Polars at {CASEDIR}")
    with open("output_bat", "w") as file:
        file.writelines("#!/bin/bash\n")
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
    dat: list[Any] = []
    for folder in folders:
        if folder == "m0.0000":
            continue
        if "clcd.out" in next(os.walk(folder))[2]:
            file_location: str = os.path.join(folder, "clcd.out")
            dat.append(np.loadtxt(file_location))
    df: DataFrame = pd.DataFrame(dat, columns=["AoA", "CL", "CD", "Cm"]).sort_values(
        "AoA",
    )
    df.to_csv("clcd.f2w", index=False)
    os.chdir(HOMEDIR)
    return df


def make_polars_bash(
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