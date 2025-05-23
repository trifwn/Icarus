import logging
import os

import numpy as np
from pandas import DataFrame

from ICARUS.airfoils import Airfoil
from ICARUS.core.types import FloatArray
from ICARUS.database import Database


def save_multiple_reyn(
    airfoil: Airfoil,
    polars: list[dict[str, FloatArray]],
    reynolds: list[float],
) -> None:
    DB = Database.get_instance()
    airfoil_dir: str = os.path.join(DB.DB2D, f"{airfoil.name.upper()}")
    for i, reyn_data in enumerate(polars):
        if len(reyn_data) == 0:
            continue
        try:
            os.chdir(airfoil_dir)
        except FileNotFoundError:
            try:
                os.makedirs(airfoil_dir, exist_ok=True)
            except FileExistsError:
                pass
            os.chdir(airfoil_dir)

        reyndir: str = f"Reynolds_{np.format_float_scientific(reynolds[i], sign=False, precision=3, min_digits=3).replace('+', '')}"
        os.makedirs(reyndir, exist_ok=True)
        os.chdir(reyndir)
        df: DataFrame = DataFrame(reyn_data).T.rename(
            columns={"index": "AoA", 0: "CL", 1: "CD", 2: "Cm"},
        )
        # Check if the DataFrame is empty by checking if the CL column is empty
        if df["CL"].empty:
            print(f"Reynolds {reynolds[i]} failed to converge to a solution")
            continue

        fname = "clcd.xfoil"
        df.to_csv(fname, sep="\t", index=True, index_label="AoA")
    # If the airfoil doesn't exist in the DB, save it
    files_in_folder = os.listdir(airfoil_dir)
    if airfoil.file_name in files_in_folder:
        logging.info(
            f"{airfoil.name.upper()} already exists in the database. We do not need to save it.",
        )
    else:
        airfoil.save_selig(airfoil_dir)

    # Add Results to Database
    print(f"Adding {airfoil.name.upper()} to the database")
    DB.load_airfoil_data(airfoil)
