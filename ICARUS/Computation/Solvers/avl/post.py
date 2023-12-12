import os

import numpy as np
import pandas
from pandas import DataFrame

from ICARUS.Computation.Solvers.AVL import AVL_HOME
from ICARUS.Core.types import FloatArray
from ICARUS.Database import DB
from ICARUS.Database import DB3D
from ICARUS.Database.utils import angle_to_case
from ICARUS.Database.utils import disturbance_to_case
from ICARUS.Flight_Dynamics.state import State
from ICARUS.Vehicle.plane import Airplane


def process_avl_angle_run(RESULTS_DIR: str, plane: Airplane, angles: FloatArray) -> DataFrame:
    """POST-PROCESSING OF POLAR RUNS - RETURNS AN ARRAY WITH THE FOLLOWING ORDER OF VECTORS: AOA,CL,CD,CM

    Args:
        PLANEDIR (str): Path to plane directory
        plane (Airplane): Airplane object
        angles (FloatArray): Array of angles of attack

    Returns:
        FloatArray: Array with the following order of vectors: AOA,CL,CD,CM
    """

    AoAs = []
    CLs = []
    CDs = []
    Cms = []
    RESULTS_DIR = os.path.join(DB3D, plane.directory, "AVL")
    for angle in angles:
        file = os.path.join(RESULTS_DIR, f"{angle_to_case(angle)}.txt")

        with open(file, encoding="utf-8") as f:
            con = f.readlines()

        CL = con[23]
        CD = con[24]
        Cm = con[20]

        AoAs.append(angle)
        if CL[11] == "-":
            CLs.append(float(CL[11:19]))
        else:
            CLs.append(float(CL[12:19]))
        if CD[11] == "-":
            CDs.append(float(CD[11:19]))
        else:
            CDs.append(float(CD[12:19]))
        if Cm[33] == "-":
            Cms.append(float(Cm[33:41]))
        else:
            Cms.append(float(Cm[34:41]))
    polar_df = DataFrame(
        np.array([AoAs, CLs, CDs, Cms]).T,
        columns=["AoA", "AVL CL", "AVL CD", "AVL Cm"],
    ).reset_index(drop=True)
    file_2_save = os.path.join(DB3D, plane.directory, "polars.avl")
    DB.vehicles_db.polars[plane.name] = polar_df
    polar_df.to_csv(file_2_save)
    return polar_df


# POST PROCESSING OF FINITE DIFFERENCE RUNS - RETURNS PERTURBATION ANALYSIS RESULTS IN THE FORM OF DICTIONARIES
# - NECESSARY FOR FINITE-DIFFERENCES BASED DYNAMIC ANALYSIS
# USEFUL FOR SENSITIVITY ANALYSIS WITH RESPECT TO STATE VARIABLE INCREMENTS


def finite_difs_post(plane: Airplane, state: State):
    DYNDIR = os.path.join(DB3D, plane.name, "AVL", "Dynamics")

    pertrubation_df: DataFrame = DataFrame()
    CZ_li = []
    CX_li = []
    Cm_li = []
    CY_li = []
    Cl_li = []
    Cn_li = []

    inc_dict = {
        "axes": "notyet",
        "var": f"{var}",
        "vals": np.concatenate((-inc_ar, inc_ar)),
    }

    for dst in state.disturbances:
        casefile = disturbance_to_case(dst)
        with open(casefile, encoding='utf-8') as f:
            lines = f.readlines()

    if var in ["w", "u", "q"]:
        for inc_v in inc_ar:
            bp = os.path.join(DYNDIR, f"dif_b_{var}_{inc_v}")
            bf = open(bp)
            con = bf.readlines()

            temp_CZ = con[21]
            temp_CX = con[19]
            temp_Cm = con[20]

            CZ_li.append(float(temp_CZ[11:19]))

            CX_li.append(float(temp_CX[11:19]))

            if temp_Cm[33] == "-":
                Cm_li.append(float(temp_Cm[33:41]))
            else:
                Cm_li.append(float(temp_Cm[34:41]))

            bf.close()

        for inc_v in inc_ar:
            fp = os.path.join(DYNDIR, f"dif_f_{var}_{inc_v}")
            ff = open(fp)
            con = ff.readlines()

            temp_CZ = con[21]
            temp_CX = con[19]
            temp_Cm = con[20]

            CZ_li.append(float(temp_CZ[11:19]))

            CX_li.append(float(temp_CX[11:19]))

            if temp_Cm[33] == "-":
                Cm_li.append(float(temp_Cm[33:41]))
            else:
                Cm_li.append(float(temp_Cm[34:41]))

            ff.close()

            inc_dict["axes"] = "longitudinal"
            inc_dict["CX"] = np.array(CX_li)
            inc_dict["CZ"] = np.array(CZ_li)
            inc_dict["Cm"] = np.array(Cm_li)
    else:
        for inc_v in inc_ar:
            bp = os.path.join(DYNDIR, f"dif_b_{var}_{inc_v}")
            bf = open(bp)
            con = bf.readlines()

            temp_CY = con[20]
            temp_Cl = con[19]
            temp_Cn = con[21]

            CY_li.append(float(temp_CY[11:19]))

            Cl_li.append(float(temp_Cl[33:41]))

            if temp_Cn[33] == "-":
                Cn_li.append(float(temp_Cn[33:41]))
            else:
                Cn_li.append(float(temp_Cn[34:41]))

            bf.close()

        for inc_v in inc_ar:
            fp = os.path.join(DYNDIR, f"dif_f_{var}_{inc_v}")
            ff = open(fp)
            con = ff.readlines()

            temp_CY = con[20]
            temp_Cl = con[19]
            temp_Cn = con[21]

            CY_li.append(float(temp_CY[11:19]))

            Cl_li.append(float(temp_Cl[33:41]))

            if temp_Cn[33] == "-":
                Cn_li.append(float(temp_Cn[33:41]))
            else:
                Cn_li.append(float(temp_Cn[34:41]))

            ff.close()

            inc_dict["axes"] = "lateral"
            inc_dict["CY"] = np.array(CY_li)
            inc_dict["Cl"] = np.array(Cl_li)
            inc_dict["Cn"] = np.array(Cn_li)

    return inc_dict
