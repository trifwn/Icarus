import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

GNVP_LOGGER = logging.getLogger("ICARUS.solvers.GenuVP")


def get_loads_convergence(file: str, gnvp_version: int) -> DataFrame:
    if gnvp_version == 3:
        return get_loads_convergence_3(file)
    if gnvp_version == 7:
        GNVP_LOGGER.info("Load Convergence for GenuVP7 not implemented")
        df = DataFrame()
        return df
    raise ValueError(f"GenuVP version {gnvp_version} not recognized")


def get_error_convergence(file: str, df: DataFrame, gnvp_version: int) -> DataFrame:
    if gnvp_version == 3:
        return get_error_convergence_3(file, df)
    if gnvp_version == 7:
        GNVP_LOGGER.info("Error Convergence for GenuVP7 not implemented")
        df = DataFrame()
        return df
    raise ValueError(f"GenuVP version {gnvp_version} not recognized")


# GET THE SCANNING FROM THE DATABASE AND MAKE DF WITH IT
def get_loads_convergence_3(file: str) -> DataFrame:
    df = DataFrame()
    try:
        df = pd.read_csv(file, sep=r"\s+", names=cols)
    except Exception as e:
        print(f"Cant Load {file} as Loads_Aero.dat!\nGot error {e}")
    finally:
        return df


def get_error_convergence_3(file: str, df: DataFrame) -> DataFrame:
    try:
        with open(file, encoding="UTF-8") as f:
            lines: list[str] = f.readlines()
        time: list[int] = []
        error: list[float] = []
        errorm: list[float] = []
        for line in lines:
            if not line.startswith(" STEP="):
                continue

            a: list[str] = line[6:].split()
            time.append(int(a[0]))
            error.append(float(a[2]))
            errorm.append(float(a[6]))
        try:
            foo: int = len(df["TTIME"])
            if foo > len(time):
                df = df.tail(len(time))
            else:
                error = error[-foo:]
                errorm = errorm[-foo:]
            df.insert(3, "ERROR", np.array(error, dtype=float))
            df.insert(3, "ERRORM", np.array(errorm, dtype=float))
        except ValueError as e:
            print(e)
            GNVP_LOGGER.info(
                f"Some Run Had Problems! Could not add convergence data\n{e}",
            )

    except FileNotFoundError:
        GNVP_LOGGER.info(f"No gnvp3.out or gnvp7.out file found in {file}!")
    return df


cols: list[str] = [
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
