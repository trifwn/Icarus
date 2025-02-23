from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from pandas import DataFrame
from pandas import Series

from ICARUS.core.types import FloatArray
from ICARUS.database.db import Database
from ICARUS.database.utils import disturbance_to_case
from ICARUS.flight_dynamics.state import State
from ICARUS.vehicle.plane import Airplane


def log_forces(CASEDIR: str, HOMEDIR: str, gnvp_version: int) -> DataFrame:
    """Convert the forces to polars and return a dataframe with them.

    Args:
        CASEDIR (str): Case Directory
        HOMEDIR (str): Home Directory
        gnvp_version(int): Version of GNVP

    Returns:
        DataFrame: Resulting Polars

    """
    folders: list[str] = next(os.walk(CASEDIR))[1]
    pols: list[list[float]] = []
    for folder in folders:
        folder_path = os.path.join(CASEDIR, folder)
        files: list[str] = next(os.walk(folder_path))[2]
        if "LOADS_aer.dat" in files:
            name = float("".join(c for c in folder if (c.isdigit() or c == ".")))
            file_name = os.path.join(folder_path, "LOADS_aer.dat")
            dat: FloatArray = np.loadtxt(file_name)[-1]
            if folder.startswith("m"):
                a: list[float] = [-name, *dat]
            else:
                a = [name, *dat]
            pols.append(a)
    if gnvp_version == 7:
        cols = cols_7
    elif gnvp_version == 3:
        cols = cols_3
    else:
        raise ValueError(f"GenuVP version {gnvp_version} does not exist")

    # Create the DataFrame
    df: DataFrame = DataFrame(pols, columns=cols)
    df.pop("TIME")
    df.pop("PSI")
    df = df.sort_values("AoA").reset_index(drop=True)
    df = rotate_gnvp_forces(df, df["AoA"], gnvp_version)
    df = set_default_name_to_use(df, gnvp_version, default_name_to_use="Potential")
    df = df[["AoA"] + [col for col in df.columns if col != "AoA"]]

    # Save the DataFrame
    forces_file: str = os.path.join(CASEDIR, f"forces.gnvp{gnvp_version}")
    df.to_csv(forces_file, index=False, float_format="%.10f")
    os.chdir(HOMEDIR)
    return df


def forces_to_pertrubation_results(
    plane: Airplane,
    state: State,
    gnvp_version: int,
    default_name_to_use: str | None = None,
) -> DataFrame:
    DB = Database.get_instance()

    DYNDIR = DB.get_vehicle_case_directory(
        airplane=plane,
        state=state,
        solver=f"GenuVP{gnvp_version}",
        case="Dynamics",
    )

    pols = []
    for dst in state.disturbances:
        case = disturbance_to_case(dst)
        case_path = os.path.join(DYNDIR, case)
        casefiles: list[str] = next(os.walk(case_path))[2]
        if "LOADS_aer.dat" in casefiles:
            loads_file = os.path.join(case_path, "LOADS_aer.dat")
            dat: FloatArray = np.loadtxt(loads_file)[-1]

            pols.append([dst.var, dst.amplitude if dst.amplitude else 0, *dat])

    # Parse the File
    if gnvp_version == 7:
        cols = cols_7
    elif gnvp_version == 3:
        cols = cols_3
    else:
        raise ValueError(f"GenuVP version {gnvp_version} does not exist")

    # Create the DataFrame
    df: DataFrame = DataFrame(pols, columns=["Type", "Epsilon", *cols[1:]])

    for col in df.columns:
        if col.endswith("Fz") or col.endswith("Fx") or col.endswith("Mx") or col.endswith("Mz"):
            df[col] = -df[col]

    df.pop("TIME")
    df.pop("PSI")
    df = df.sort_values("Type").reset_index(drop=True)
    if default_name_to_use is not None:
        name = default_name_to_use
    else:
        name = "Potential"
    df = set_default_name_to_use(df, gnvp_version, default_name_to_use=name)

    perturbation_file: str = os.path.join(DYNDIR, f"pertrubations.gnvp{gnvp_version}")
    df.to_csv(perturbation_file, index=False)
    return df


def rotate_gnvp_forces(
    rawforces: DataFrame,
    alpha_deg: float | Series[float] | FloatArray,
    gnvp_version: int,
) -> DataFrame:
    data = DataFrame()
    AoA: float | Series[float] | FloatArray = alpha_deg * np.pi / 180

    for name in ["Type", "Epsilon"]:
        if name in rawforces.columns:
            data[name] = rawforces[name]

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
            logging.debug(f"Key error {e}")

    data["AoA"] = alpha_deg
    # Reindex the dataframe sort by AoA
    data = data.sort_values(by="AoA").reset_index(drop=True)
    return data


def set_default_name_to_use(forces: DataFrame, gnvp_version: int, default_name_to_use: str | None = None) -> DataFrame:
    if default_name_to_use is None:
        default_name_to_use = "Potential"

    forces["Fx"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} Fx"]
    forces["Fy"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} Fy"]
    forces["Fz"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} Fz"]
    forces["Mx"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} Mx"]
    forces["My"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} My"]
    forces["Mz"] = forces[f"GenuVP{gnvp_version} {default_name_to_use} Mz"]
    return forces


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

# cols_old: list[str] = [
#     "AoA",
#     "TTIME",
#     "PSIB",
#     "TFORC(1)",
#     "TFORC(2)",
#     "TFORC(3)",
#     "TAMOM(1)",
#     "TAMOM(2)",
#     "TAMOM(3)",
#     "TFORC2D(1)",
#     "TFORC2D(2)",
#     "TFORC2D(3)",
#     "TAMOM2D(1)",
#     "TAMOM2D(2)",
#     "TAMOM2D(3)",
#     "TFORCDS2D(1)",
#     "TFORCDS2D(2)",
#     "TFORCDS2D(3)",
#     "TAMOMDS2D(1)",
#     "TAMOMDS2D(2)",
#     "TAMOMDS2D(3)",
# ]
