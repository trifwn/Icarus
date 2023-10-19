import os
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype
from numpy import floating
from numpy import ndarray
from pandas import DataFrame
from pandas import Series


def log_forces(CASEDIR: str, HOMEDIR: str, genu_version: int) -> DataFrame:
    """
    Convert the forces to polars and return a dataframe with them.

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        genu_version(int): Version of GNVP

    Returns:
        DataFrame: Resulting Polars
    """
    os.chdir(CASEDIR)

    folders: list[str] = next(os.walk("."))[1]
    # print("Making Polars")
    pols: list[list[float]] = []
    for folder in folders:
        os.chdir(os.path.join(CASEDIR, folder))
        files: list[str] = next(os.walk("."))[2]
        if "LOADS_aer.dat" in files:
            name = float("".join(c for c in folder if (c.isdigit() or c == ".")))
            dat: ndarray[Any, dtype[floating[Any]]] = np.loadtxt("LOADS_aer.dat")[-1]
            if folder.startswith("m"):
                a: list[float] = [-name, *dat]
            else:
                a = [name, *dat]
            pols.append(a)
        os.chdir(f"{CASEDIR}")
    df: DataFrame = DataFrame(pols, columns=cols)
    df.pop("TTIME")
    df.pop("PSIB")
    df = df.sort_values("AoA").reset_index(drop=True)
    df.to_csv(f"forces.gnvp{genu_version}", index=False)
    os.chdir(HOMEDIR)
    # df = rotate_forces(df, df['AoA'])
    return df


def forces_to_pertrubation_results(DYNDIR: str, HOMEDIR: str) -> DataFrame:
    os.chdir(DYNDIR)
    folders: list[str] = next(os.walk("."))[1]
    print("Logging Pertrubations")
    pols: list[list[float | str]] = []
    for folder in folders:
        os.chdir(os.path.join(DYNDIR, folder))
        files: list[str] = next(os.walk("."))[2]
        if "LOADS_aer.dat" in files:
            dat: ndarray[Any, dtype[floating[Any]]] = np.loadtxt("LOADS_aer.dat")[-1]
            if folder == "Trim":
                pols.append([0, str(folder), *dat])
                continue

            # RECONSTRUCT NAME
            value: str = ""
            name: str = ""
            flag = False
            for c in folder[1:]:
                if (c != "_") and (not flag):
                    value += c
                elif c == "_":
                    flag = True
                else:
                    name += c
            value_num: float = float(value)
            if folder.startswith("m"):
                value_num = -value_num

            pols.append([value_num, name, *dat])
            os.chdir(os.path.join(DYNDIR, folder))
        os.chdir(f"{DYNDIR}")

    df: DataFrame = DataFrame(pols, columns=["Epsilon", "Type", *cols[1:]])
    df.pop("TTIME")
    df.pop("PSIB")
    df = df.sort_values("Type").reset_index(drop=True)
    df.to_csv("pertrubations.genu", index=False)
    os.chdir(HOMEDIR)
    return df


def rotate_forces(
    rawpolars: DataFrame,
    alpha_deg: float | Series | ndarray[Any, dtype[floating[Any]]],
    preferred: str = "2D",
    save: bool = False,
) -> DataFrame:
    data = pd.DataFrame()
    AoA: float | Series[float] | ndarray[Any, dtype[floating[Any]]] = alpha_deg * np.pi / 180

    for enc, name in zip(["", "2D", "DS2D"], ["Potential", "2D", "ONERA"]):
        f_x: Series[Any] = rawpolars[f"TFORC{enc}(1)"]
        f_y: Series[Any] = rawpolars[f"TFORC{enc}(2)"]
        f_z: Series[Any] = rawpolars[f"TFORC{enc}(3)"]

        m_x: Series[Any] = rawpolars[f"TAMOM{enc}(1)"]
        m_y: Series[Any] = rawpolars[f"TAMOM{enc}(2)"]
        m_z: Series[Any] = rawpolars[f"TAMOM{enc}(3)"]

        f_x_rot: Series[Any] = f_x * np.cos(-AoA) - f_z * np.sin(-AoA)
        f_y_rot: Series[Any] = f_y
        f_z_rot: Series[Any] = f_x * np.sin(-AoA) + f_z * np.cos(-AoA)

        m_x_rot: Series[Any] = m_x * np.cos(-AoA) - m_z * np.sin(-AoA)
        m_y_rot: Series[Any] = m_y
        m_z_rot: Series[Any] = m_x * np.sin(-AoA) + m_z * np.cos(-AoA)

        data[f"Fx_{name}"] = f_x_rot
        data[f"Fy_{name}"] = f_y_rot
        data[f"Fz_{name}"] = f_z_rot
        data[f"L_{name}"] = m_x_rot
        data[f"M_{name}"] = m_y_rot
        data[f"N_{name}"] = m_z_rot

    data["AoA"] = alpha_deg
    # print(f"Using {preferred} polars")
    data["Fx"] = data[f"Fx_{preferred}"]
    data["Fy"] = data[f"Fy_{preferred}"]
    data["Fz"] = data[f"Fz_{preferred}"]
    data["L"] = data[f"L_{preferred}"]
    data["M"] = data[f"M_{preferred}"]
    data["N"] = data[f"N_{preferred}"]
    # Reindex the dataframe sort by AoA
    data = data.sort_values(by="AoA").reset_index(drop=True)
    return data


cols: list[str] = [
    "AoA",
    "TTIME",
    "PSIB",
    "TFORC(1)",
    "TFORC(2)",
    "TFORC(3)",
    "TAMOM(1)",
    "TAMOM(2)",
    "TAMOM(3)",
    "TFORC2D(1)",
    "TFORC2D(2)",
    "TFORC2D(3)",
    "TAMOM2D(1)",
    "TAMOM2D(2)",
    "TAMOM2D(3)",
    "TFORCDS2D(1)",
    "TFORCDS2D(2)",
    "TFORCDS2D(3)",
    "TAMOMDS2D(1)",
    "TAMOMDS2D(2)",
    "TAMOMDS2D(3)",
]
