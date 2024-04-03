import os
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.database import DB
from ICARUS.database.database2D import Database_2D
from ICARUS.database.database3D import Database_3D


def read_polars_2d(XFLRdir: str) -> None:
    """
    Reads the polars from XFLR5 and stores them in the database.

    Args:
        DB (Database_2D | Database): Database Object
        XFLRdir (str): XFLR directory
    """
    HOMEDIR: str = DB.HOMEDIR
    foils_db: Database_2D = DB.foils_db
    try:
        os.chdir(XFLRdir)
    except FileNotFoundError:
        print("XFLR5 Directory Not Found!")
        os.makedirs(XFLRdir, exist_ok=True)
        return
    directories: list[str] = next(os.walk("."))[1]
    for airf in directories:
        if airf == "XFLs":
            continue
        pattern = r"\([^)]*\)|[^0-9a-zA-Z]+"
        cleaned_string = re.sub(pattern, " ", airf)
        # Split the cleaned string into numeric and text parts
        foil: str = "".join(filter(str.isdigit, cleaned_string))
        text_part: str = "".join(filter(str.isalpha, cleaned_string))

        if text_part.find("flap") != -1:
            name: str = f"{foil + 'fl'}"
        else:
            name = foil

        if airf.startswith("NACA"):
            name = "NACA" + name
        if name not in foils_db._raw_data.keys():
            foils_db._raw_data[name] = {}

        if airf.startswith("NACA"):
            os.chdir(airf)
            directory_files: list[str] = next(os.walk("."))[2]
            for file in directory_files:
                if file.startswith("NACA") and file.endswith(".txt"):
                    # get the 7th line of the file
                    with open(file) as f:
                        line: str = f.readlines()[7]
                    # get the reynolds number from the format:
                    # We should get the string after Re = and Before Ncrit
                    #  Mach =   0.000     Re =     3.000 e 6     Ncrit =   9.000
                    reyn_str: str = line.split("Re =")[1].split("Ncrit")[0]
                    # remove spaces
                    reyn_str = re.sub(r"\s+", "", reyn_str)
                    # convert to float
                    reyn: float = float(reyn_str)
                    try:
                        dat: DataFrame = pd.read_csv(
                            file,
                            sep="  ",
                            header=None,
                            skiprows=11,
                            engine="python",
                        )
                    except pd.errors.EmptyDataError:
                        print(f"Error")
                        continue

                    dat.columns = pd.Index(xfoilcols, dtype="str")
                    dat = dat.drop(
                        [
                            "CDp",
                            "Top Xtr",
                            "Bot Xtr",
                            "Cpmin",
                            "Chinge",
                            "_",
                            "_",
                            "XCp",
                        ],
                        axis=1,
                    )
                    if "XFLR" not in foils_db._raw_data[name].keys():
                        foils_db._raw_data[name]["XFLR"] = {}

                    reyn_str = np.format_float_scientific(reyn, sign=False, precision=3, min_digits=3).replace(
                        "+",
                        "",
                    )
                    foils_db._raw_data[name]["XFLR"][reyn_str] = dat
                    dat = pd.read_csv(
                        file,
                        sep="  ",
                        header=None,
                        skiprows=11,
                        engine="python",
                    )
                    dat.columns = pd.Index(xfoilcols)
                    dat = dat.drop(
                        [
                            "CDp",
                            "Top Xtr",
                            "Bot Xtr",
                            "Cpmin",
                            "Chinge",
                            "_",
                            "_",
                            "XCp",
                        ],
                        axis=1,
                    )
                    if "XFLR" not in foils_db._raw_data[name].keys():
                        foils_db._raw_data[name]["XFLR"] = {}

                    reyn_str = np.format_float_scientific(reyn, sign=False, precision=3, min_digits=3).replace(
                        "+",
                        "",
                    )
                    foils_db._raw_data[name]["XFLR"][reyn_str] = dat

                    dat = pd.read_csv(
                        file,
                        sep="  ",
                        header=None,
                        skiprows=11,
                        engine="python",
                    )
                    dat.columns = pd.Index(xfoilcols)
                    dat = dat.drop(
                        [
                            "CDp",
                            "Top Xtr",
                            "Bot Xtr",
                            "Cpmin",
                            "Chinge",
                            "_",
                            "_",
                            "XCp",
                        ],
                        axis=1,
                    )
                    if "XFLR" not in foils_db._raw_data[name].keys():
                        foils_db._raw_data[name]["XFLR"] = {}

                    reyn_str = np.format_float_scientific(reyn, sign=False, precision=3, min_digits=3).replace("+", "")
                    foils_db._raw_data[name]["XFLR"][reyn_str] = dat
            os.chdir(XFLRdir)
    os.chdir(HOMEDIR)


def read_polars_3d(
    filename: str,
    plane_name: str,
) -> DataFrame | None:
    """
    Reads the plane polars (3d) from XFLR5 and stores them in the database.

    Args:
        filename (str): Plane polar filename
        plane_name (str): Planename

    Returns:
        DataFrame | None: _description_
    """
    if f"XFLR_{plane_name}" not in DB.vehicles_db.polars.keys():
        # import csv into pandas Dataframe and skip first 7 rows
        df: DataFrame = pd.read_csv(
            filename,
            skiprows=7,
            delim_whitespace=True,
            on_bad_lines="skip",
        )
        # rename columns
        df.rename(columns={"alpha": "AoA"}, inplace=True)

        # convert to float
        df = df.astype(float)
        DB.vehicles_db.polars[f"XFLR_{plane_name}"] = df
        return df
    else:
        print("Polar Already Exists!")
        return None


xfoilcols: list[str] = [
    "AoA",
    "CL",
    "CD",
    "CDp",
    "Cm",
    "Top Xtr",
    "Bot Xtr",
    "Cpmin",
    "Chinge",
    "_",
    "_",
    "XCp",
]
