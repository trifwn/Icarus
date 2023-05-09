import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.Database.Database_2D import Database_2D


def readPolars2D(db: Database_2D, XFLRdir: str) -> None:

    HOMEDIR = db.HOMEDIR
    os.chdir(XFLRdir)
    directories: list[str] = next(os.walk("."))[1]
    for airf in directories:
        if airf not in db.data.keys():
            db.data[airf] = {}
        if airf.startswith("NACA"):
            os.chdir(airf)
            directory_files: list[str] = next(os.walk("."))[2]
            for polar in directory_files:
                if polar.startswith("NACA"):
                    foo: list[str] = polar[4:].split("_")
                    reyn: float = float(foo[2][2:]) * 1e6
                    dat: DataFrame = pd.read_csv(
                        polar,
                        sep="  ",
                        header=None,
                        skiprows=11,
                        engine="python",
                    )
                    dat.columns = xfoilcols
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
                    if "XFLR" not in db.data[airf].keys():
                        db.data[airf]["XFLR"] = {}
                    reyn_str: str = np.format_float_scientific(
                        reyn,
                        sign=False,
                        precision=3,
                    ).replace("+", "")
                    db.data[airf]["XFLR"][reyn_str] = dat
            os.chdir(XFLRdir)
    os.chdir(HOMEDIR)


def readPolars3D(db, FILENAME, name):
    if f"XFLR_{name}" not in db.data.keys():
        # import csv into pandas Dataframe and skip first 7 rows
        df = pd.read_csv(
            FILENAME,
            skiprows=7,
            delim_whitespace=True,
            on_bad_lines="skip",
        )
        # rename columns
        df.rename(columns={"alpha": "AoA"}, inplace=True)

        # convert to float
        df = df.astype(float)
        db.data[f"XFLR_{name}"] = df
        return df
    else:
        print("Polar Already Exists!")


xfoilcols = [
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
