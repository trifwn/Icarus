import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from ICARUS.computation.solvers.OpenFoam.post_process.get_aero_coefficients import (
    get_coefficients,
)


def make_polars(case_directory: str, angles: list[float]) -> DataFrame:
    """Function to make polars from OpenFoam results

    Args:
        case_directory (str): Path to the case directory containing the results
        angles (list[float]): Angles to make polars for
    Returns:
        DataFrame: Dataframe Containing CL, CD, CM for all angles

    """
    cd: list[float] = []
    cl: list[float] = []
    cm: list[float] = []
    folders: list[str] = next(os.walk(case_directory))[1]
    angles_succeded: list[float] = []

    for angle in angles:
        if angle >= 0:
            folder: str = str(angle)[::-1].zfill(7)[::-1]
        else:
            folder = "m" + str(angle)[::-1].strip("-").zfill(6)[::-1]

        if folder in folders:
            data = get_coefficients(case_directory, angle)
            if data is not None:
                (
                    Time,
                    Cd,
                    Cdf,
                    Cdr,
                    Cl,
                    Clf,
                    Clr,
                    CmPitch,
                    CmRoll,
                    CmYaw,
                    Cs,
                    Csf,
                    Csr,
                ) = (float(i) for i in data.split("\t"))
                angles_succeded.append(angle)
                cd.append(Cd)
                cl.append(Cl)
                cm.append(CmPitch)
    df: DataFrame = pd.DataFrame(
        np.vstack([angles_succeded, cl, cd, cm]).T,
        columns=["AoA", "CL", "CD", "CM"],
    ).sort_values("AoA")

    clcd_file = os.path.join(case_directory, "clcd.of")
    df.to_csv(clcd_file, index=False)
    return df
