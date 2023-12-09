import os
import stat
import subprocess
from os import stat_result
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.Core.types import FloatArray


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
    res_file = "AERLOAD.OUT"
    for folder in folders:
        if "AERLOAD.OUT" in next(os.walk(folder))[2]:
            with open(f"{folder}/{res_file}", encoding="UTF-8") as f:
                data: str = f.read()
            if data == "":
                continue
            values = [x.strip() for x in data.split(" ") if x != ""]
            cl, cd, cm, aoa = (values[7], values[8], values[11], values[17])
            with open(f"{folder}/clcd.out", "w", encoding="UTF-8") as f:
                f.write(f"{aoa}\t{cl}\t{cd}\t{cm}")
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
) -> FloatArray:
    """
    Make the polars from the forces and return an np array with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
    Returns:
        FloatArray: Polars
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
    clcd: FloatArray = np.array(nums)
    os.chdir(HOMEDIR)
    return clcd
