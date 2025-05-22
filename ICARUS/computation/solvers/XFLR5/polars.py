import os
import re

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.airfoils import AirfoilData
from ICARUS.database import Database


def parse_airfoil_name(file_name: str) -> str:
    """
    Parses the airfoil name from the filename.
    An example: NACA 0008_T1_Re0.015_M0.00_N9.0_XtrTop10%_XtrBot10%.dat
    The function should return NACA0008
    """
    file_name = file_name.replace(" ", "")
    name: str = file_name.split("_")[0]
    return name


def read_polars_2d(directory: str) -> None:
    """Reads the polars from XFLR5 and stores them in the database.

    Args:
        DB (Database_2D | Database): Database Object
        XFLRdir (str): XFLR directory

    """
    DB = Database.get_instance()

    # Check if the XFLR5 directory exists
    if not os.path.exists(directory):
        print("XFLR5 directory not found")
        return

    files: list[str] = next(os.walk(directory))[2]
    reynolds_data = {}
    for file in files:
        airfoil_name: str = parse_airfoil_name(file)
        file_name = os.path.join(directory, file)
        # get the 7th line of the file
        with open(file_name) as f:
            line: str = f.readlines()[7]
        # get the reynolds number from the format:
        # We should get the string after Re = and Before Ncrit
        #  Mach =   0.000     Re =     3.000 e 6     Ncrit =   9.000
        reyn_str: str = line.split("Re =")[1].split("Ncrit")[0]
        # remove spaces
        reyn_str = re.sub(r"\s+", "", reyn_str)
        # convert to float
        reyn: float = float(reyn_str)
        reyn_str = np.format_float_scientific(
            reyn,
            sign=False,
            precision=3,
            min_digits=3,
        ).replace("+", "")
        try:
            xflr_df: DataFrame = pd.read_csv(
                file_name,
                sep="  ",
                header=None,
                skiprows=11,
                engine="python",
            )
        except pd.errors.EmptyDataError:
            print("\tCould not read the file")
            continue

        xflr_df.columns = pd.Index(xfoilcols)
        xflr_df = xflr_df.drop(
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
        if airfoil_name not in reynolds_data.keys():
            reynolds_data[airfoil_name] = {}
        if reyn_str not in reynolds_data[airfoil_name].keys():
            reynolds_data[airfoil_name][reyn_str] = xflr_df

        reynolds_data[airfoil_name][reyn_str] = xflr_df

    for airfoil_name in reynolds_data.keys():
        # Add to the database
        if airfoil_name not in DB.foils_db.polars.keys():
            DB.foils_db.polars[airfoil_name] = AirfoilData(
                name=airfoil_name,
                data={"XFLR": reynolds_data[airfoil_name]},
            )
        else:
            DB.foils_db.polars[airfoil_name].add_solver("XFLR", reynolds_data[airfoil_name])


def read_XFLR5_polars(
    filename: str,
    plane_name: str,
) -> DataFrame | None:
    """Reads the plane polars (3d) from XFLR5 and stores them in the database.

    Args:
        filename (str): Plane polar filename
        plane_name (str): Planename

    Returns:
        DataFrame | None: _description_

    """
    DB = Database.get_instance()
    # import csv into pandas Dataframe and skip first 7 rows
    df: DataFrame = pd.read_csv(
        filename,
        skiprows=7,
        sep=r"\s+",
        on_bad_lines="skip",
    )
    # rename columns
    df.rename(columns={"alpha": "AoA"}, inplace=True)

    # Renamce CL, CD, Cm columns
    df.rename(
        columns={
            "CL": "XFLR5 CL",
            "CD": "XFLR5 CD",
            "Cm": "XFLR5 Cm",
        },
        inplace=True,
    )
    # Keep only the relevant columns and drop the rest
    df = df[["AoA", "XFLR5 CL", "XFLR5 CD", "XFLR5 Cm"]]
    df = df.astype(float)
    df = df.reindex(columns=["AoA", "XFLR5 CL", "XFLR5 CD", "XFLR5 Cm"])
    # Sort the dataframe by AoA
    df = df.sort_values(by="AoA")

    if plane_name not in DB.vehicles_db.polars.keys():
        DB.vehicles_db.polars[plane_name] = df
    else:
        # Check if the XFLR Key is already in the database and drop the old one
        cols = [col for col in DB.vehicles_db.polars[plane_name].columns if "XFLR5" in col]
        DB.vehicles_db.polars[plane_name].drop(columns=cols, inplace=True)

        DB.vehicles_db.polars[plane_name] = DB.vehicles_db.polars[plane_name].merge(
            df,
            on="AoA",
            how="outer",
        )
    return df


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
