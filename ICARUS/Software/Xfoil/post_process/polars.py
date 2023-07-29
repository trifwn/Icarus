import os

import numpy as np
from pandas import DataFrame

from ICARUS.Airfoils.airfoilD import AirfoilD
from ICARUS.Core.types import FloatArray
from ICARUS.Database.Database_2D import Database_2D


def save_multiple_reyn(
    db: Database_2D,
    airfoil: AirfoilD,
    polars: list[dict[str, FloatArray]],
    reynolds: list[float],
) -> None:
    airfoil_dir: str = os.path.join(db.DATADIR, f"NACA{airfoil.name}")

    for i, reyn_data in enumerate(polars):
        try:
            os.chdir(airfoil_dir)
        except FileNotFoundError:
            os.makedirs(airfoil_dir, exist_ok=True)
            os.chdir(airfoil_dir)

        reyndir: str = f"Reynolds_{np.format_float_scientific(reynolds[i],sign=False,precision=3).replace('+', '')}"
        os.makedirs(reyndir, exist_ok=True)
        os.chdir(reyndir)

        df = DataFrame(reyn_data).T.rename(
            columns={"index": "AoA", 0: "CL", 1: "CD", 2: "CM"},
        )
        fname = "clcd.xfoil"
        df.to_csv(fname, sep="\t", index=True, index_label="AoA")


def save_multiple_airfoils_reyn(
    airfoils: str,
    polars: list[list[dict[str, FloatArray]]],
    reynolds: list[float],
) -> None:
    masterDir: str = os.getcwd()
    os.chdir(masterDir)
    for airfoil, clcd_data in zip(airfoils, polars):
        os.chdir(masterDir)
        os.chdir(os.path.join("Data", "2D", f"NACA{airfoil}"))
        airfoilPath: str = os.getcwd()

        for i, reyn_data in enumerate(clcd_data):
            os.chdir(airfoilPath)

            reyndir: str = f"Reynolds_{np.format_float_scientific(reynolds[i],sign=False,precision=3).replace('+', '')}"
            os.makedirs(reyndir, exist_ok=True)
            os.chdir(reyndir)
            cwd: str = os.getcwd()

            for angle in reyn_data.keys():
                os.chdir(cwd)
                if float(angle) >= 0:
                    folder: str = str(angle)[::-1].zfill(7)[::-1]
                else:
                    folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]
                os.makedirs(folder, exist_ok=True)
                os.chdir(folder)
                fname = "clcd.xfoil"
                with open(fname, "w") as file:
                    pols = str(angle)
                    for i in reyn_data[angle]:
                        pols += f"\t{str(i)}"
                    file.writelines(pols)
