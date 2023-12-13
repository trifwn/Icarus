import os
from typing import Any
from venv import logger

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series

from ICARUS.Core.types import FloatArray
from ICARUS.Flight_Dynamics.state import State


def log_forces(CASEDIR: str, HOMEDIR: str, gnvp_version: int) -> DataFrame:
    """
    Convert the forces to polars and return a dataframe with them.

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        gnvp_version(int): Version of GNVP

    Returns:
        DataFrame: Resulting Polars
    """
    GNVPDIR = os.path.join(CASEDIR, f"GenuVP{gnvp_version}")
    print(f"CASEDIR ")
    os.chdir(GNVPDIR)

    folders: list[str] = next(os.walk("."))[1]
    # print("Making Polars")
    pols: list[list[float]] = []
    for folder in folders:
        os.chdir(os.path.join(GNVPDIR, folder))
        files: list[str] = next(os.walk("."))[2]
        if "LOADS_aer.dat" in files:
            name = float("".join(c for c in folder if (c.isdigit() or c == ".")))
            dat: FloatArray = np.loadtxt("LOADS_aer.dat")[-1]
            if folder.startswith("m"):
                a: list[float] = [-name, *dat]
            else:
                a = [name, *dat]
            pols.append(a)
        os.chdir(f"{GNVPDIR}")
    if gnvp_version == 7:
        cols = cols_7
    elif gnvp_version == 3:
        cols = cols_3
    else:
        raise ValueError(f"GenuVP version {gnvp_version} does not exist")

    df: DataFrame = DataFrame(pols, columns=cols)
    df.pop("TIME")
    df.pop("PSI")
    df = df.sort_values("AoA").reset_index(drop=True)
    df = rotate_gnvp_forces(df, df["AoA"], gnvp_version)
    df = df.sort_values("AoA").reset_index(drop=True)
    forces_file: str = os.path.join(CASEDIR, f"forces.gnvp{gnvp_version}")
    df.to_csv(forces_file, index=False, float_format="%.10f")
    os.chdir(HOMEDIR)
    return df


def forces_to_pertrubation_results(DYNDIR: str, HOMEDIR: str, state: State, gnvp_version: int) -> DataFrame:
    os.chdir(DYNDIR)
    folders: list[str] = next(os.walk("."))[1]
    print("Logging Pertrubations")
    pols: list[list[float | str]] = []
    for folder in folders:
        os.chdir(os.path.join(DYNDIR, folder))
        files: list[str] = next(os.walk("."))[2]
        if "LOADS_aer.dat" in files:
            dat: FloatArray = np.loadtxt("LOADS_aer.dat")[-1]
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

    if gnvp_version == 7:
        cols = cols_7
    elif gnvp_version == 3:
        cols = cols_3
    else:
        raise ValueError(f"GenuVP version {gnvp_version} does not exist")
    df: DataFrame = DataFrame(pols, columns=["Epsilon", "Type", *cols[1:]])
    df.pop("TTIME")
    df.pop("PSIB")
    df = df.sort_values("Type").reset_index(drop=True)
    df = rotate_gnvp_forces(df, state.trim["AoA"], gnvp_version)

    df.to_csv(f"pertrubations.gnvp{gnvp_version}", index=False)
    os.chdir(HOMEDIR)
    return df


def rotate_gnvp_forces(
    rawforces: DataFrame,
    alpha_deg: float | Series | FloatArray,
    gnvp_version: int,
    default_name_to_use: str = "2D",
) -> DataFrame:
    data = DataFrame()
    AoA: float | Series[float] | FloatArray = alpha_deg * np.pi / 180

    name = None
    for name in ["Potential", "2D", "ONERA"]:
        try:
            f_x: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} Fx"]
            f_y: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} Fy"]
            f_z: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} Fz"]

            m_x: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} Mx"]
            m_y: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} My"]
            m_z: Series[Any] = rawforces[f"GenuVP{gnvp_version} {name} Mz"]

            f_x_rot: Series[Any] = f_x * np.cos(-AoA) - f_z * np.sin(-AoA)
            f_y_rot: Series[Any] = f_y
            f_z_rot: Series[Any] = f_x * np.sin(-AoA) + f_z * np.cos(-AoA)

            m_x_rot: Series[Any] = m_x * np.cos(-AoA) - m_z * np.sin(-AoA)
            m_y_rot: Series[Any] = m_y
            m_z_rot: Series[Any] = m_x * np.sin(-AoA) + m_z * np.cos(-AoA)

            data[f"GenuVP{gnvp_version} {name} Fx"] = f_x_rot
            data[f"GenuVP{gnvp_version} {name} Fy"] = f_y_rot
            data[f"GenuVP{gnvp_version} {name} Fz"] = f_z_rot
            data[f"GenuVP{gnvp_version} {name} Mx"] = m_x_rot
            data[f"GenuVP{gnvp_version} {name} My"] = m_y_rot
            data[f"GenuVP{gnvp_version} {name} Mz"] = m_z_rot
        except KeyError as e:
            logger.debug(f"Key error {e}")

    data["AoA"] = alpha_deg
    data["Fx"] = data[f"GenuVP{gnvp_version} {default_name_to_use} Fx"]
    data["Fy"] = data[f"GenuVP{gnvp_version} {default_name_to_use} Fy"]
    data["Fz"] = data[f"GenuVP{gnvp_version} {default_name_to_use} Fz"]
    data["L"] = data[f"GenuVP{gnvp_version} {default_name_to_use} Mx"]
    data["M"] = data[f"GenuVP{gnvp_version} {default_name_to_use} My"]
    data["N"] = data[f"GenuVP{gnvp_version} {default_name_to_use} Mz"]
    # Reindex the dataframe sort by AoA
    data = data.sort_values(by="AoA").reset_index(drop=True)
    return data


cols_old: list[str] = [
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

cols_3: list[str] = [
    "AoA",
    "TIME",
    "PSI",
    "GenuVP3 Potential Fx",
    "GenuVP3 Potential Fy",
    "GenuVP3 Potential Fz",
    "GenuVP3 Potential Mx",
    "GenuVP3 Potential My",
    "GenuVP3 Potential Mz",
    "GenuVP3 2D Fx",
    "GenuVP3 2D Fy",
    "GenuVP3 2D Fz",
    "GenuVP3 2D Mx",
    "GenuVP3 2D My",
    "GenuVP3 2D Mz",
    "GenuVP3 ONERA Fx",
    "GenuVP3 ONERA Fy",
    "GenuVP3 ONERA Fz",
    "GenuVP3 ONERA Mx",
    "GenuVP3 ONERA My",
    "GenuVP3 ONERA Mz",
]

cols_7: list[str] = [
    "AoA",
    "TIME",
    "PSI",
    "GenuVP7 Potential Fx",
    "GenuVP7 Potential Fy",
    "GenuVP7 Potential Fz",
    "GenuVP7 Potential Mx",
    "GenuVP7 Potential My",
    "GenuVP7 Potential Mz",
    "GenuVP7 2D Fx",
    "GenuVP7 2D Fy",
    "GenuVP7 2D Fz",
    "GenuVP7 2D Mx",
    "GenuVP7 2D My",
    "GenuVP7 2D Mz",
    "GenuVP7 ONERA Fx",
    "GenuVP7 ONERA Fy",
    "GenuVP7 ONERA Fz",
    "GenuVP7 ONERA Mx",
    "GenuVP7 ONERA My",
    "GenuVP7 ONERA Mz",
]
