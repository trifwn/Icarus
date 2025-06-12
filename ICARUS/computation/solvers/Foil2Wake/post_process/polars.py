import os

import pandas as pd
from pandas import DataFrame


def make_polars(case_directory: str, HOMEDIR: str) -> DataFrame:
    """Make the polars from the forces and return a dataframe with them

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
    Returns:
        DataFrame: Dataframe Containing CL, CD, CM for all angles

    """
    folders: list[str] = next(os.walk(case_directory))[1]
    res_file = "AERLOAD.OUT"
    dat: list[list[float]] = []
    for folder in folders:
        folder_path = os.path.join(case_directory, folder)
        if "AERLOAD.OUT" in next(os.walk(folder))[2]:
            load_file = os.path.join(folder_path, res_file)
            with open(load_file, encoding="UTF-8") as f:
                data: str = f.read()
            if data == "":
                continue
            values = [x.strip() for x in data.split(" ") if x != ""]
            cl, cd, cm, aoa = (values[7], values[8], values[11], values[17])
            dat.append([float(aoa), float(cl), float(cd), float(cm)])

    df: DataFrame = pd.DataFrame(dat, columns=["AoA", "CL", "CD", "Cm"]).sort_values(
        "AoA",
    )
    clcd_file = os.path.join(case_directory, "clcd.f2w")
    df.to_csv(clcd_file, index=False)
    return df
