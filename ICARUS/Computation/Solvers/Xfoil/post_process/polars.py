import os

import numpy as np
from pandas import DataFrame

from ICARUS.Airfoils.airfoil import Airfoil
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB


def save_multiple_reyn(
    airfoil: Airfoil,
    polars: list[dict[str, FloatArray]],
    reynolds: list[float],
) -> None:
    airfoil_dir: str = os.path.join(DB.foils_db.DATADIR, f"NACA{airfoil.name}")
    for i, reyn_data in enumerate(polars):
        if len(reyn_data) == 0:
            continue
        try:
            os.chdir(airfoil_dir)
        except FileNotFoundError:
            os.makedirs(airfoil_dir, exist_ok=True)
            os.chdir(airfoil_dir)

        reyndir: str = (
            f"Reynolds_{np.format_float_scientific(reynolds[i],sign=False,precision=3, min_digits=3).replace('+', '')}"
        )
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
    airfoil.save_selig_te(airfoil_dir)
