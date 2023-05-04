import os

import numpy as np
import pandas as pd


def readPolars2D(db, XFLRdir):
    HOMEDIR = db.HOMEDIR
    os.chdir(XFLRdir)
    files = next(os.walk("."))[1]
    for airf in files:
        if airf not in db.Data.keys():
            db.Data[airf] = {}
        if airf.startswith("NACA"):
            os.chdir(airf)
            dat = next(os.walk("."))[2]
            for polar in dat:
                if polar.startswith("NACA"):
                    foo = polar[4:].split("_")
                    reyn = float(foo[2][2:]) * 1e6
                    dat = pd.read_csv(
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
                    if "XFLR" not in db.Data[airf].keys():
                        db.Data[airf]["XFLR"] = {}
                    reyn = np.format_float_scientific(
                        reyn,
                        sign=False,
                        precision=3,
                    ).replace("+", "")
                    db.Data[airf]["XFLR"][reyn] = dat
            os.chdir(XFLRdir)
    os.chdir(HOMEDIR)


def readPolars3D(db, FILENAME, name):
    if f"XFLR_{name}" not in db.Data.keys():
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
        db.Data[f"XFLR_{name}"] = df
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
